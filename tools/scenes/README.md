# tools/scenes: moving 3D test sequences for the SR / RR references

A rendered `sponza16` is in the private storage repo (`HANDOFF.md`); restore it instead of rendering if it fits.

Path-traced sequences rendered with Blender's Cycles: real sampling noise, exact motion vectors and every
guide RR reads (depth, normals, diffuse / specular albedo, roughness), plus converged ground truth at the
output size. Nothing here is committed except the scripts: Blender, the assets and every render live
under `build/`.

## Setup (once)

```bash
# Blender 4.2.3 LTS (download.blender.org answers 403 to curl; any official mirror works).
mkdir -p build/blender && cd build/blender
curl -sSfL -A Mozilla/5.0 -o blender.tar.xz \
  https://mirrors.ocf.berkeley.edu/blender/release/Blender4.2/blender-4.2.3-linux-x64.tar.xz
echo "3a64efd1982465395abab4259b4091d5c8c56054c7267e9633e4f702a71ea3f4  blender.tar.xz" | sha256sum -c
tar xf blender.tar.xz && cd ../..

# Khronos glTF sample assets (Sponza: Cryengine Limited License; DamagedHelmet: CC-BY 4.0). Not committed.
mkdir -p build/scenes && cd build/scenes
for m in Sponza DamagedHelmet; do
  mkdir -p $m
  curl -sSfL "https://api.github.com/repos/KhronosGroup/glTF-Sample-Assets/contents/Models/$m/glTF" \
    | python3 -c "import json,sys; [print(e['download_url']) for e in json.load(sys.stdin)]" > $m/urls.txt
  (cd $m && xargs -n1 -P8 curl -sSfLO < urls.txt)
done
cd ../..

# GPU rendering: Cycles HIP needs the ROCm runtime, which SteamOS lacks, so Blender runs inside the
# ROCm container with Blender's X libraries added. Its first render compiles the gfx1201 kernel with
# the container's hipcc (~70 s); build/blender/kcache keeps it.
podman run --name rb docker.io/rocm/dev-ubuntu-24.04:7.2.4 bash -c 'apt-get update -qq && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libxrender1 libxi6 libxxf86vm1 libxfixes3 \
  libxkbcommon0 libsm6 libgl1 libegl1 libx11-6 libxext6 libwayland-client0'
podman commit rb localhost/rocm-blender:1 && podman rm rb
mkdir -p build/blender/kcache
```

## Render, convert, check

```bash
B=build/blender/blender-4.2.3-linux-x64/blender
podman run --rm --device /dev/kfd --device /dev/dri --group-add keep-groups --security-opt label=disable \
  -v $PWD:/w -v $PWD/build/blender/kcache:/root/.cache/cycles localhost/rocm-blender:1 bash -c "cd /w && \
  $B -b -P tools/scenes/render_seq.py -- build/scenes/sponza16 --sponza build/scenes/Sponza \
     --helmet build/scenes/DamagedHelmet --frames 16 --render 960 540 --output 1920 1080 \
     --noisy-spp 2 --clean-spp 256 --gt-spp 512 --device HIP"
$B -b -P tools/scenes/convert_seq.py -- build/scenes/sponza16        # host Blender is enough here
build/venv/bin/python -I tools/scenes/check_mv.py build/scenes/sponza16/sr
```

Without `--device HIP` (and without the container) it renders on the CPU, several times slower.
Rendering skips EXRs that already exist, so an interrupted run resumes.

`build/scenes/SEQ/` then holds `rr/` (rrref frames: noisy colour + guides), `sr/` (srref frames: the clean
aliased colour, depth, motion) and `gt/gt_i.f32` (RGBA32F ground truth). Run the references on them as
usual and score with the ground-truth directory in place of a picture:

```bash
build/venv/bin/python -I tools/rr/eval_rr.py build/scenes/sponza16/gt build/scenes/sponza16/rr OUT
build/venv/bin/python -I tools/sr/eval_sr.py build/scenes/sponza16/gt build/scenes/sponza16/sr OUT
```

## Conventions (as NGX expects them; `convert_seq.py`)

- Jitter: NGX `Jitter.Offset`, Halton(2,3); render pixel (x, y) samples the scene at pixel centre minus
  the jitter. Done with the camera shift (units of the larger image side; +y up), point-sampled
  (`filter_width` 0.01). The ground truth is unjittered, anti-aliased and denoised.
- Motion vectors: render pixels, current pixel to its previous-frame position, y down, without jitter
  (the camera shift is not animated, so Cycles' vector pass does not see it). `check_mv.py` confirms the
  sign by warping the previous frame.
- Depth: hardware reverse-Z, `near / view depth` (near 0.1), `depth_inverted`.
- Normals world space; albedos Cycles' DiffCol / GlossCol; roughness an AOV wired to every material's
  Principled BSDF roughness input.

## Notes

- Speed on the RX 9070 XT (Blender 4.2.3, HIP): ~49 s per frame, of which ~43 s is the 512-spp 1080p ground
  truth (software BVH traversal; 4.2 does not use the RT units here). `--gt-spp 128` is ~4x faster and the
  denoiser hides most of the difference; `render.use_persistent_data` would also skip the per-render
  scene sync (not set yet).
- Point-sampled frames keep fireflies (`sponza16` frame 6 has one of 8688): they inflate `check_mv.py`'s
  error for the two pairs that include that frame (the sign test still wins elsewhere).
