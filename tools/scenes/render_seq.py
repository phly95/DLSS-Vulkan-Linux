# Render a moving 3D test sequence for the SR / RR references with Blender's Cycles path tracer.
#
#   blender -b -P tools/scenes/render_seq.py -- OUT_DIR --sponza DIR --helmet DIR \
#       [--render 960 540] [--output 1920 1080] [--frames 16] [--noisy-spp 2] [--clean-spp 64] [--gt-spp 256]
#
# Scene: Khronos's glTF Sponza with the DamagedHelmet flying and spinning through it, a camera dollying
# and panning, sun + sky light (path-traced, so real noise at low sample counts). Per frame i it writes
# OUT_DIR/exr/{noisy,clean,gt}_i.exr (multilayer, linear):
#   noisy: render size, --noisy-spp samples, point-sampled at the pixel centre minus NGX's jitter, with
#          every guide pass (depth, normal, vector, diffuse / glossy colour, roughness AOV);
#   clean: the same pixels at --clean-spp (an aliased but converged frame, what SR gets from a game);
#   gt:    output size, --gt-spp, anti-aliased and denoised (the reference picture).
# tools/scenes/convert_seq.py turns them into the rrref / srref input files.
#
# The assets are not part of the repository (Sponza is under Crytek's licence); see tools/scenes/README.md.
import argparse
import json
import math
import os
import random
import sys

import bpy
from mathutils import Vector


def halton(i, b):
    f, r = 1.0, 0.0
    while i > 0:
        f /= b
        r += f * (i % b)
        i //= b
    return r


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--sponza", required=True)
    ap.add_argument("--helmet", required=True)
    ap.add_argument("--render", type=int, nargs=2, default=[960, 540])
    ap.add_argument("--output", type=int, nargs=2, default=[1920, 1080])
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--noisy-spp", type=int, default=2)
    ap.add_argument("--clean-spp", type=int, default=64)
    ap.add_argument("--gt-spp", type=int, default=256)
    ap.add_argument("--device", choices=["CPU", "HIP"], default="CPU",
                    help="HIP: the GPU through ROCm (run Blender inside a ROCm container; tools/scenes/README.md)")
    ap.add_argument("--only", choices=["noisy", "clean", "gt"], nargs="*")
    ap.add_argument("--seed", type=int, default=None, help="seed for random camera/helmet paths")
    ap.add_argument("--reset-frame", type=int, default=None, help="frame index with reset=1")
    ap.add_argument("--no-gt", action="store_true", help="skip ground-truth renders")
    return ap.parse_args(argv)


def import_gltf(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    return [o for o in bpy.data.objects if o not in before]


def roughness_aov():
    """An AOV 'rough' in every material: the Principled BSDF's roughness input, linked or constant."""
    for m in bpy.data.materials:
        if not m.use_nodes:
            continue
        nt = m.node_tree
        for n in list(nt.nodes):
            if n.type != "BSDF_PRINCIPLED":
                continue
            aov = nt.nodes.new("ShaderNodeOutputAOV")
            aov.aov_name = "rough"
            src = n.inputs["Roughness"]
            if src.is_linked:
                nt.links.new(src.links[0].from_socket, aov.inputs["Value"])
            else:
                aov.inputs["Value"].default_value = src.default_value
            break


def main():
    a = args()
    os.makedirs(os.path.join(a.out, "exr"), exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene

    sponza = import_gltf(os.path.join(a.sponza, "Sponza.gltf"))
    helmet = import_gltf(os.path.join(a.helmet, "DamagedHelmet.gltf"))
    roughness_aov()
    # the helmet's root (the importer parents meshes under the glTF node hierarchy)
    hroot = [o for o in helmet if o.parent is None][0]
    hroot.scale = (0.6, 0.6, 0.6)

    # camera & helmet paths: default dolly or seeded random paths within Sponza atrium
    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = 24
    cam_data.clip_start = 0.1
    cam = bpy.data.objects.new("cam", cam_data)
    sc.collection.objects.link(cam)
    sc.camera = cam

    if a.seed is not None:
        rng = random.Random(a.seed)
        reset_frame = a.reset_frame if a.reset_frame is not None else rng.randint(4, min(8, max(4, a.frames - 1)))
        cx0 = rng.uniform(-10.5, -9.0)
        cvx = rng.uniform(0.06, 0.12)
        cy0 = rng.uniform(-0.4, 0.4)
        cvy = rng.uniform(-0.02, 0.02)
        cz0 = rng.uniform(1.5, 1.9)
        cvz = rng.uniform(-0.01, 0.01)
        cpitch = math.radians(rng.uniform(84.0, 88.0))
        cyaw0 = rng.uniform(-94.0, -90.0)
        cyaw_rate = rng.uniform(0.3, 0.7)

        hx0 = rng.uniform(-4.5, -3.5)
        hvx = rng.uniform(0.02, 0.05)
        hy0 = rng.uniform(1.5, 2.0)
        hvy = rng.uniform(-0.25, -0.18)
        hz0 = rng.uniform(1.4, 1.7)
        h_amp = rng.uniform(0.1, 0.2)
        h_freq = rng.uniform(0.3, 0.5)
        hrot0 = rng.uniform(15.0, 25.0)
        hrot_rate = rng.uniform(4.0, 6.0)

        for f in range(a.frames + 2):
            t = f - 1
            cam.location = (cx0 + cvx * t, cy0 + cvy * t, cz0 + cvz * t)
            cam.rotation_euler = (cpitch, 0.0, math.radians(cyaw0 + cyaw_rate * t))
            cam.keyframe_insert("location", frame=f)
            cam.keyframe_insert("rotation_euler", frame=f)

            hroot.location = (hx0 + hvx * t, hy0 + hvy * t, hz0 + h_amp * math.sin(h_freq * t))
            hroot.rotation_euler = (math.radians(90), 0.0, math.radians(hrot0 + hrot_rate * t))
            hroot.keyframe_insert("location", frame=f)
            hroot.keyframe_insert("rotation_euler", frame=f)
    else:
        reset_frame = a.reset_frame if a.reset_frame is not None else -1
        for f in range(a.frames + 2):
            t = f - 1                                   # keyframes one frame before and after: exact vectors
            cam.location = (-10.0 + 0.08 * t, 0.3 + 0.01 * t, 1.7 + 0.01 * t)
            cam.rotation_euler = (math.radians(86), 0.0, math.radians(-92 + 0.5 * t))
            cam.keyframe_insert("location", frame=f)
            cam.keyframe_insert("rotation_euler", frame=f)
            hroot.location = (-4.0 + 0.03 * t, 1.8 - 0.22 * t, 1.6 + 0.15 * math.sin(0.4 * t))
            hroot.rotation_euler = (math.radians(90), 0.0, math.radians(20 + 5.0 * t))
            hroot.keyframe_insert("location", frame=f)
            hroot.keyframe_insert("rotation_euler", frame=f)
    for o in [cam, hroot]:
        for fc in o.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"

    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = 8.0
    sun.angle = math.radians(1.0)
    so = bpy.data.objects.new("sun", sun)
    so.rotation_euler = (math.radians(35), math.radians(10), math.radians(60))
    sc.collection.objects.link(so)
    world = bpy.data.worlds.new("sky")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.45, 0.6, 0.9, 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.5
    sc.world = world

    sc.render.engine = "CYCLES"
    cy = sc.cycles
    if a.device == "HIP":
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "HIP"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = d.type == "HIP"
        cy.device = "GPU"
    else:
        cy.device = "CPU"
    cy.max_bounces = 4
    cy.use_adaptive_sampling = False
    cy.use_denoising = False
    sc.render.use_motion_blur = False
    sc.render.film_transparent = False
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
    sc.render.image_settings.color_depth = "32"
    sc.render.image_settings.exr_codec = "ZIP"
    vl = sc.view_layers[0]
    aov = vl.aovs.add()
    aov.name = "rough"
    aov.type = "VALUE"

    rw, rh = a.render
    meta = dict(render=a.render, output=a.output, frames=a.frames, near=cam_data.clip_start,
                seed=a.seed, reset_frame=reset_frame, jitter=[])
    for i in range(a.frames):
        sc.frame_set(i + 1)
        jx, jy = halton(i + 1, 2) - 0.5, halton(i + 1, 3) - 0.5
        meta["jitter"].append([jx, jy])
        for kind in ("noisy", "clean", "gt"):
            if a.no_gt and kind == "gt":
                continue
            if a.only and kind not in a.only:
                continue
            path = os.path.join(a.out, "exr", f"{kind}_{i}.exr")
            if os.path.exists(path):
                continue
            gt = kind == "gt"
            sc.render.resolution_x, sc.render.resolution_y = (a.output if gt else a.render)
            # NGX: render pixel (x, y) (y down) samples the picture at pixel centre minus the jitter.
            # Camera shift is in units of the larger sensor dimension; +x moves the view right, +y up.
            cam_data.shift_x = 0.0 if gt else -jx / max(rw, rh)
            cam_data.shift_y = 0.0 if gt else jy / max(rw, rh)
            cy.samples = {"noisy": a.noisy_spp, "clean": a.clean_spp, "gt": a.gt_spp}[kind]
            cy.seed = 1000 * i + {"noisy": 1, "clean": 2, "gt": 3}[kind]
            cy.filter_width = 1.5 if gt else 0.01      # point sampling for the game-like frames
            cy.use_denoising = gt
            for p in ("use_pass_z", "use_pass_normal", "use_pass_vector", "use_pass_diffuse_color",
                      "use_pass_glossy_color"):
                setattr(vl, p, not gt)
            sc.render.filepath = path
            bpy.ops.render.render(write_still=True)
            print(f"wrote {path}", flush=True)
    with open(os.path.join(a.out, "sequence.json"), "w") as f:
        json.dump(meta, f, indent=1)


main()
