bl_info = {
    "name": "Gaussian Splat Generator",
    "author": "Generated Addon",
    "version": (2, 0, 0),
    "blender": (5, 1, 0),
    "location": "View3D > Sidebar > GSplat",
    "description": "Renders cameras distributed uniformly across a sphere for 3D Gaussian Splatting",
    "category": "Render",
}

import bpy
import os
import math
from bpy.props import IntProperty, FloatProperty, StringProperty, BoolProperty, EnumProperty
from bpy.types import Panel, Operator, PropertyGroup


# ---------------------------------------------------------------------------
# Fibonacci sphere camera positions
# ---------------------------------------------------------------------------

def fibonacci_sphere_positions(n, radius,
                               top_angle_deg=0.0, bottom_angle_deg=0.0,
                               rot_x_deg=0.0, rot_y_deg=0.0, rot_z_deg=0.0):
    """
    Return a list of n (x, y, z) positions uniformly distributed on a sphere
    of the given radius using the Fibonacci / golden-angle method.

    top_angle_deg / bottom_angle_deg – degrees from each pole to exclude.
    rot_x/y/z_deg – Euler XYZ rotation (degrees) applied to the whole sphere.
    """
    import mathutils

    positions = []
    golden = math.pi * (3.0 - math.sqrt(5.0))   # ~137.5° golden angle
    y_max  = math.cos(math.radians(max(0.0, top_angle_deg)))
    y_min  = -math.cos(math.radians(max(0.0, bottom_angle_deg)))
    span   = y_max - y_min
    if n < 1 or span <= 0.0:
        return positions
    denom = float(n - 1) if n > 1 else 1.0

    rot_mat = mathutils.Euler(
        (math.radians(rot_x_deg), math.radians(rot_y_deg), math.radians(rot_z_deg)), 'XYZ'
    ).to_matrix()

    for i in range(n):
        y     = y_max - (i / denom) * span
        r     = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden * i
        v = rot_mat @ mathutils.Vector((
            radius * r * math.cos(theta),
            radius * r * math.sin(theta),
            radius * y,
        ))
        positions.append((v.x, v.y, v.z))
    return positions


# ---------------------------------------------------------------------------
# Module-level render state
# ---------------------------------------------------------------------------

_state = {
    "active":       False,
    "stop":         False,
    "frame_index":  0,
    "total_frames": 0,
    "positions":      [],   # list of (x, y, z) camera positions
    "colmap_frames":  [],
    "colmap_pending": False,  # True while waiting to write the point cloud
    # Saved scene state
    "original_camera":           None,
    "original_output":           "",
    "original_frame":            1,
    "original_world":            None,
    "original_film_transparent": False,
    "original_color_mode":       "RGB",
    "original_lens":             50.0,
    # Blender objects created by us
    "camera_obj":  None,
    "camera_data": None,
    "target_obj":  None,
}

_overlay_handle = None
_overlay_cache  = {"shader": None, "batch": None, "params": None}


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def _cleanup():
    """Restore scene state and remove temporary objects."""
    scene = bpy.context.scene
    props = scene.gsplat_props

    props.is_rendering   = False
    props.status_message = ""
    props.progress       = 0.0
    _state["active"]         = False
    _state["colmap_pending"] = False

    if _state["original_camera"] is not None:
        scene.camera = _state["original_camera"]

    scene.render.filepath = _state["original_output"]
    scene.render.film_transparent = _state["original_film_transparent"]
    scene.render.image_settings.color_mode = _state["original_color_mode"]

    if _state["original_world"] is not None:
        scene.world = _state["original_world"]
        _state["original_world"] = None

    scene.frame_set(_state["original_frame"])

    for key in ("camera_obj", "target_obj"):
        obj = _state.get(key)
        if obj and obj.name in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        _state[key] = None

    if _state["camera_data"] and _state["camera_data"].name in bpy.data.cameras:
        bpy.data.cameras.remove(_state["camera_data"])
    _state["camera_data"] = None


# ---------------------------------------------------------------------------
# All-cameras viewport overlay
# ---------------------------------------------------------------------------

def _tag_viewport_redraw():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def _build_camera_overlay_batch(camera_count, radius, lens, aspect,
                                top_angle_deg=0.0, bottom_angle_deg=0.0,
                                rot_x_deg=0.0, rot_y_deg=0.0, rot_z_deg=0.0):
    """
    Build a GPU line batch of camera frustums for every Fibonacci sphere position.
    Each frustum is drawn as: near rectangle + far rectangle + 4 apex→far-corner edges.
    FOV is derived from the given lens (mm) assuming a 36 mm full-frame sensor.
    """
    import mathutils
    import gpu
    from gpu_extras.batch import batch_for_shader

    positions = fibonacci_sphere_positions(
        camera_count, radius, top_angle_deg, bottom_angle_deg,
        rot_x_deg, rot_y_deg, rot_z_deg,
    )
    near_dist = radius * 0.02
    far_dist  = radius * 0.09
    # Half-angle tangents from a 36 mm full-frame sensor
    tan_h = (36.0 / 2.0) / max(lens, 0.001)
    tan_v = tan_h / max(aspect, 0.001)

    ref_a  = mathutils.Vector((0.0, 0.0, 1.0))
    ref_b  = mathutils.Vector((1.0, 0.0, 0.0))
    target = mathutils.Vector((0.0, 0.0, 0.0))

    coords = []
    for (x, y, z) in positions:
        cam_pos = mathutils.Vector((x, y, z))
        fwd     = (target - cam_pos).normalized()
        ref     = ref_b if abs(fwd.dot(ref_a)) > 0.9 else ref_a
        right   = fwd.cross(ref).normalized()
        up      = right.cross(fwd).normalized()

        rects = []
        for dist in (near_dist, far_dist):
            c  = cam_pos + fwd * dist
            hw = dist * tan_h
            hv = dist * tan_v
            corners = (
                c + up * hv - right * hw,   # TL
                c + up * hv + right * hw,   # TR
                c - up * hv + right * hw,   # BR
                c - up * hv - right * hw,   # BL
            )
            rects.append(corners)
            # Rectangle outline
            for i in range(4):
                coords += [corners[i][:], corners[(i + 1) % 4][:]]

        # 4 frustum edges: apex → far corners
        for corner in rects[1]:
            coords += [cam_pos[:], corner[:]]

    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    batch  = batch_for_shader(shader, 'LINES', {"pos": coords})
    return shader, batch


def _draw_all_cameras_callback():
    import gpu
    try:
        scene = bpy.context.scene
        props = scene.gsplat_props
    except Exception:
        return
    if not props.show_all_cameras:
        return

    render = scene.render
    scale  = render.resolution_percentage / 100.0
    rw     = render.resolution_x * scale
    rh     = render.resolution_y * scale
    aspect = rw / rh if rh else 1.0

    params = (props.camera_count, props.radius, props.lens, round(aspect, 4),
              round(props.top_exclusion_angle, 2), round(props.bottom_exclusion_angle, 2),
              round(props.sphere_rot_x, 2), round(props.sphere_rot_y, 2), round(props.sphere_rot_z, 2))
    if _overlay_cache["batch"] is None or _overlay_cache["params"] != params:
        _overlay_cache["shader"], _overlay_cache["batch"] = \
            _build_camera_overlay_batch(
                props.camera_count, props.radius, props.lens, aspect,
                props.top_exclusion_angle, props.bottom_exclusion_angle,
                props.sphere_rot_x, props.sphere_rot_y, props.sphere_rot_z,
            )
        _overlay_cache["params"] = params

    shader = _overlay_cache["shader"]
    batch  = _overlay_cache["batch"]

    gpu.state.blend_set('ALPHA')
    shader.bind()
    shader.uniform_float("color", (1.0, 0.75, 0.0, 0.85))
    batch.draw(shader)
    gpu.state.blend_set('NONE')


def _register_camera_overlay():
    global _overlay_handle
    if _overlay_handle is None:
        _overlay_handle = bpy.types.SpaceView3D.draw_handler_add(
            _draw_all_cameras_callback, (), 'WINDOW', 'POST_VIEW'
        )
    _tag_viewport_redraw()


def _unregister_camera_overlay():
    global _overlay_handle
    if _overlay_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_overlay_handle, 'WINDOW')
        _overlay_handle = None
    _overlay_cache["shader"] = None
    _overlay_cache["batch"]  = None
    _overlay_cache["params"] = None
    _tag_viewport_redraw()


# ---------------------------------------------------------------------------
# Mask writer
# ---------------------------------------------------------------------------

def _write_mask(filepath_no_ext, render_settings, masks_dir):
    """
    Extract the alpha channel of the rendered image as a greyscale mask PNG.
    White = opaque subject, Black = transparent background.
    Filename: <basename><ext>.png  (COLMAP / LichtFeld convention).
    """
    ext_map = {
        'PNG': '.png', 'JPEG': '.jpg', 'JPEG2000': '.jp2',
        'TIFF': '.tif', 'TARGA': '.tga', 'TARGA_RAW': '.tga',
        'BMP': '.bmp', 'OPEN_EXR': '.exr', 'OPEN_EXR_MULTILAYER': '.exr',
        'HDR': '.hdr', 'CINEON': '.cin', 'DPX': '.dpx', 'WEBP': '.webp',
    }
    fmt      = render_settings.image_settings.file_format
    ext      = ext_map.get(fmt, '.png')
    src_path = filepath_no_ext + ext
    if not os.path.exists(src_path):
        return

    src = bpy.data.images.load(src_path, check_existing=False)
    src.alpha_mode = 'STRAIGHT'
    w, h   = src.size[0], src.size[1]
    pixels = list(src.pixels)
    bpy.data.images.remove(src)

    mask    = bpy.data.images.new("_GSplat_Mask", width=w, height=h, alpha=False)
    mask_px = [0.0] * (w * h * 4)
    for i in range(w * h):
        v = 1.0 if pixels[i * 4 + 3] > 0.0 else 0.0
        mask_px[i * 4]     = v
        mask_px[i * 4 + 1] = v
        mask_px[i * 4 + 2] = v
        mask_px[i * 4 + 3] = 1.0
    mask.pixels[:] = mask_px

    basename  = os.path.basename(filepath_no_ext)
    mask_path = os.path.join(masks_dir, basename + ext + ".png")
    mask.filepath_raw = mask_path
    mask.file_format  = 'PNG'
    mask.save()
    bpy.data.images.remove(mask)


# ---------------------------------------------------------------------------
# COLMAP helpers
# ---------------------------------------------------------------------------

def _blender_camera_to_colmap(cam_obj):
    """World-to-camera pose in COLMAP convention (X right, Y down, Z forward)."""
    import mathutils
    depsgraph = bpy.context.evaluated_depsgraph_get()
    w2c = cam_obj.evaluated_get(depsgraph).matrix_world.inverted()
    R   = w2c.to_3x3()
    t   = w2c.translation
    R_col = mathutils.Matrix((
        ( R[0][0],  R[0][1],  R[0][2]),
        (-R[1][0], -R[1][1], -R[1][2]),
        (-R[2][0], -R[2][1], -R[2][2]),
    ))
    t_col = mathutils.Vector((t[0], -t[1], -t[2]))
    q = R_col.to_quaternion()
    return q.w, q.x, q.y, q.z, t_col.x, t_col.y, t_col.z


def _intrinsics(cam_data, render):
    scale = render.resolution_percentage / 100.0
    w = render.resolution_x * scale
    h = render.resolution_y * scale
    fit = cam_data.sensor_fit
    if fit == 'AUTO':
        fit = 'HORIZONTAL' if w >= h else 'VERTICAL'
    fx = fy = ((cam_data.lens / cam_data.sensor_width) * w
               if fit == 'HORIZONTAL'
               else (cam_data.lens / cam_data.sensor_height) * h)
    return fx, fy, w / 2.0, h / 2.0, int(w), int(h)


def _ext_for_format(fmt):
    return {
        'PNG': '.png', 'JPEG': '.jpg', 'JPEG2000': '.jp2',
        'TIFF': '.tif', 'TARGA': '.tga', 'TARGA_RAW': '.tga',
        'BMP': '.bmp', 'OPEN_EXR': '.exr', 'OPEN_EXR_MULTILAYER': '.exr',
        'HDR': '.hdr', 'CINEON': '.cin', 'DPX': '.dpx', 'WEBP': '.webp',
    }.get(fmt, '.png')


def _collect_face_sample_points(max_points=1000):
    """
    Distribute max_points random samples across all visible mesh faces,
    weighted by face area so larger faces receive proportionally more points.
    Each sample is placed at a uniformly random position on its triangle
    using the square-root barycentric method.

    Performance notes:
    - Triangle vertices are stored as raw Python floats (not mathutils.Vectors)
      to keep memory low for high-polygon scenes.
    - random.choices() is used for area-weighted selection — it is implemented
      in C, builds a cumulative-weight table once, and binary-searches per draw,
      avoiding the O(n log n) sort of the previous largest-remainder approach.
    """
    import random

    skip = {"_GSplat_Target", "_GSplat_Camera",
            "_GSplat_Preview_Target", "_GSplat_Preview_Camera"}
    depsgraph = bpy.context.evaluated_depsgraph_get()

    # Collect triangles as flat float tuples — avoids keeping Vector objects alive.
    # Layout: (ax, ay, az, bx, by, bz, cx, cy, cz, area)
    tris    = []
    weights = []
    for obj in bpy.context.scene.objects:
        if obj.name in skip or obj.hide_render or obj.type != 'MESH':
            continue
        obj_eval = obj.evaluated_get(depsgraph)
        mesh     = obj_eval.to_mesh()
        if mesh is None:
            continue
        mat = obj.matrix_world
        for poly in mesh.polygons:
            verts = [mat @ mesh.vertices[vi].co for vi in poly.vertices]
            for i in range(1, len(verts) - 1):
                A, B, C = verts[0], verts[i], verts[i + 1]
                area = ((B - A).cross(C - A)).length * 0.5
                if area > 1e-12:
                    tris.append((A.x, A.y, A.z, B.x, B.y, B.z, C.x, C.y, C.z))
                    weights.append(area)
        obj_eval.to_mesh_clear()

    if not tris:
        return []

    # Area-weighted selection in one C-level pass; no sort, no extra lists.
    selected = random.choices(tris, weights=weights, k=max_points)

    # Barycentric sampling — pure float arithmetic, no object creation per point.
    all_pts = []
    for (ax, ay, az, bx, by, bz, cx, cy, cz) in selected:
        r1 = random.random()
        r2 = random.random()
        sq = math.sqrt(r1)
        u  = 1.0 - sq
        v  = sq * (1.0 - r2)
        w  = sq * r2
        all_pts.append((
            u * ax + v * bx + w * cx,
            u * ay + v * by + w * cy,
            u * az + v * bz + w * cz,
        ))
    return all_pts


def _build_sparse_point_cloud(colmap_frames, cam_data, render, radius, max_points=1000):
    import mathutils

    fx, fy, cx, cy, w_px, h_px = _intrinsics(cam_data, render)

    pts_world = _collect_face_sample_points(max_points=max_points)
    if not pts_world:
        # Fallback: small sphere
        golden  = math.pi * (3.0 - math.sqrt(5.0))
        scene_r = radius * 0.5
        for i in range(60):
            y = 1.0 - (i / 59.0) * 2.0
            r = math.sqrt(max(0.0, 1.0 - y * y))
            pts_world.append((scene_r * r * math.cos(golden * i),
                              scene_r * r * math.sin(golden * i),
                              scene_r * y))

    # Pre-flatten every camera's rotation matrix and translation into plain Python
    # floats so the inner visibility loop does no mathutils object creation or
    # attribute lookups — just native float arithmetic.
    fit = cam_data.sensor_fit
    if fit == 'AUTO':
        fit = 'HORIZONTAL' if w_px >= h_px else 'VERTICAL'

    cam_flat = []   # (r00..r22, tx, ty, tz, f_px)
    for frame_data in colmap_frames:
        _, qw, qx, qy, qz, tx, ty, tz, lens = frame_data
        R = mathutils.Quaternion((qw, qx, qy, qz)).to_matrix()
        f_px = ((lens / cam_data.sensor_width)  * w_px if fit == 'HORIZONTAL'
                else (lens / cam_data.sensor_height) * h_px)
        cam_flat.append((
            R[0][0], R[0][1], R[0][2],
            R[1][0], R[1][1], R[1][2],
            R[2][0], R[2][1], R[2][2],
            tx, ty, tz, f_px,
        ))

    points3d           = []
    points2d_per_image = [[] for _ in colmap_frames]
    point_id           = 1
    margin             = 2
    x_hi               = w_px - margin
    y_hi               = h_px - margin

    for (X, Y, Z) in pts_world:
        track = []
        for img_idx, (r00, r01, r02,
                      r10, r11, r12,
                      r20, r21, r22,
                      tx, ty, tz, f_px_i) in enumerate(cam_flat):
            pcz = r20 * X + r21 * Y + r22 * Z + tz
            if pcz <= 0.0:
                continue
            pcx = r00 * X + r01 * Y + r02 * Z + tx
            pcy = r10 * X + r11 * Y + r12 * Z + ty
            inv_z = f_px_i / pcz
            px = pcx * inv_z + cx
            py = pcy * inv_z + cy
            if px < margin or px > x_hi or py < margin or py > y_hi:
                continue
            track.append((img_idx, px, py))
        if len(track) < 2:
            continue
        points3d.append((point_id, X, Y, Z, 128, 128, 128, 0.0, track))
        for (img_idx, px, py) in track:
            points2d_per_image[img_idx].append((px, py, point_id))
        point_id += 1

    return points3d, points2d_per_image


def write_colmap_model(output_dir, colmap_frames, cam_data, render, radius, max_points=1000):
    sparse_dir = os.path.join(output_dir, "sparse", "0")
    os.makedirs(sparse_dir, exist_ok=True)

    _, _, cx, cy, w, h = _intrinsics(cam_data, render)
    ext = _ext_for_format(render.image_settings.file_format)
    n   = len(colmap_frames)

    scale = render.resolution_percentage / 100.0
    w_px  = render.resolution_x * scale
    h_px  = render.resolution_y * scale
    fit   = cam_data.sensor_fit
    if fit == 'AUTO':
        fit = 'HORIZONTAL' if w_px >= h_px else 'VERTICAL'

    def _lens_to_fx(lens_mm):
        return ((lens_mm / cam_data.sensor_width) * w_px
                if fit == 'HORIZONTAL'
                else (lens_mm / cam_data.sensor_height) * h_px)

    points3d, points2d_per_image = _build_sparse_point_cloud(
        colmap_frames, cam_data, render, radius, max_points
    )

    # cameras.txt
    with open(os.path.join(sparse_dir, "cameras.txt"), "w", newline="\n") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {n}\n")
        for img_id, frame_data in enumerate(colmap_frames, start=1):
            lens_mm = frame_data[-1]
            f_px    = _lens_to_fx(lens_mm)
            f.write(f"{img_id} PINHOLE {w} {h} {f_px:.6f} {f_px:.6f} {cx:.6f} {cy:.6f}\n")

    # images.txt
    mean_obs = sum(len(p) for p in points2d_per_image) / max(n, 1)
    with open(os.path.join(sparse_dir, "images.txt"), "w", newline="\n") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {n}, mean observations per image: {mean_obs:.1f}\n")
        for img_id, frame_data in enumerate(colmap_frames, start=1):
            frame_idx, qw, qx, qy, qz, tx, ty, tz, _lens = frame_data
            name = f"{frame_idx:05d}{ext}"
            f.write(f"{img_id} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                    f"{tx:.9f} {ty:.9f} {tz:.9f} {img_id} {name}\n")
            pts = points2d_per_image[img_id - 1]
            if pts:
                f.write(" ".join(f"{px:.2f} {py:.2f} {pid}" for px, py, pid in pts) + "\n")
            else:
                f.write("\n")

    # points3D.txt
    mean_track = sum(len(p[8]) for p in points3d) / max(len(points3d), 1)
    with open(os.path.join(sparse_dir, "points3D.txt"), "w", newline="\n") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points3d)}, mean track length: {mean_track:.4f}\n")
        for (pid, X, Y, Z, Rc, Gc, Bc, err, track) in points3d:
            track_parts = []
            for (img_idx, px, py) in track:
                img_id    = img_idx + 1
                pt2d_list = points2d_per_image[img_idx]
                pt2d_idx  = next(
                    (i for i, (ppx, ppy, ppid) in enumerate(pt2d_list) if ppid == pid), 0
                )
                track_parts.append(f"{img_id} {pt2d_idx}")
            f.write(f"{pid} {X:.9f} {Y:.9f} {Z:.9f} {Rc} {Gc} {Bc} "
                    f"{err:.6f} {' '.join(track_parts)}\n")


# ---------------------------------------------------------------------------
# LichtFeld Studio launcher
# ---------------------------------------------------------------------------

def _launch_lichtfeld(props, output_dir):
    """
    Spawn LichtFeld Studio as a detached process, pointing it at the project
    output directory that already contains images/ and sparse/0/.
    --train starts training immediately; the GUI opens so the user can monitor.
    """
    import subprocess

    exe = bpy.path.abspath(props.lichtfeld_path).strip()
    if not exe:
        print("[GSplat] LichtFeld path is empty — skipping launch.")
        return

    # Map internal enum id → CLI value (igs_plus → igs+)
    strategy = 'igs+' if props.lichtfeld_strategy == 'igs_plus' else props.lichtfeld_strategy

    cmd = [
        exe,
        f"--data-path={output_dir}",
        f"--output-path={output_dir}",
        f"--strategy={strategy}",
        f"--iter={props.lichtfeld_iterations}",
        f"--max-cap={props.lichtfeld_max_gaussians}",
        #"--train",
    ]

    if props.transparent_background:
        # Use alpha-consistent masking; the alpha channel is used as mask by
        # default (omitting --no-alpha-as-mask keeps that behaviour).
        cmd.append("--mask-mode=alpha_consistent")
    else:
        cmd.append("--mask-mode=none")

    try:
        subprocess.Popen(cmd)
    except Exception as exc:
        print(f"[GSplat] Failed to launch LichtFeld Studio: {exc}")


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class GSplatProperties(PropertyGroup):
    camera_count: IntProperty(
        name="Camera Count",
        description="Total number of cameras distributed across the sphere",
        default=200, min=2, max=3600,
    )
    radius: FloatProperty(
        name="Radius",
        description="Distance of every camera from the origin (sphere radius)",
        default=5.0, min=0.01, soft_max=100.0, unit='LENGTH',
    )
    lens: FloatProperty(
        name="Focal Length",
        description="Camera focal length in mm",
        default=24.0, min=1.0, soft_max=135.0,
    )
    top_exclusion_angle: FloatProperty(
        name="Top Exclusion",
        description="Degrees from the north pole where no cameras are placed. "
                    "0° = cameras reach the top pole. 90° = upper hemisphere excluded",
        default=0.0, min=0.0, max=89.9,
    )
    bottom_exclusion_angle: FloatProperty(
        name="Bottom Exclusion",
        description="Degrees from the south pole where no cameras are placed. "
                    "0° = cameras reach the bottom pole. 90° = lower hemisphere excluded",
        default=0.0, min=0.0, max=89.9,
    )
    sphere_rot_x: FloatProperty(
        name="Rotate X",
        description="Rotate the entire camera sphere around the X axis (degrees)",
        default=0.0, min=-180.0, max=180.0,
    )
    sphere_rot_y: FloatProperty(
        name="Rotate Y",
        description="Rotate the entire camera sphere around the Y axis (degrees)",
        default=0.0, min=-180.0, max=180.0,
    )
    sphere_rot_z: FloatProperty(
        name="Rotate Z",
        description="Rotate the entire camera sphere around the Z axis (degrees)",
        default=0.0, min=-180.0, max=180.0,
    )
    output_path: StringProperty(
        name="Output Path",
        description="Root directory. Images → <path>/images/, COLMAP → <path>/sparse/0/",
        default="//gsplat_renders/", subtype='DIR_PATH',
    )
    is_rendering: BoolProperty(default=False)
    status_message: StringProperty(default="")
    progress: FloatProperty(default=0.0, min=0.0, max=100.0,
                            subtype='PERCENTAGE', name="Progress")
    preview_index: IntProperty(
        name="Preview Camera",
        description="Index of the Fibonacci sphere camera to preview (0-based)",
        default=0, min=0,
    )
    is_previewing: BoolProperty(default=False)
    preview_expanded: BoolProperty(default=False)
    point_cloud_points: IntProperty(
        name="Point Cloud Points",
        description="Number of random surface points sampled across all mesh faces for the COLMAP point cloud. "
                    "More points results in a better gaussian splat but may crash some systems — "
                    "values above 2,000 are rarely needed and very high values (5000+) can freeze or crash Blender",
        default=1000, min=10, max=100000,
    )
    transparent_background: BoolProperty(
        name="Transparent Background",
        description="Render the background as transparent (requires PNG or EXR output format)",
        default=True,
    )
    use_scene_camera: BoolProperty(
        name="Use Scene Camera",
        description="Render using the existing scene camera (follows its animation path) "
                    "instead of the addon's auto-positioned Fibonacci sphere cameras",
        default=False,
        update=lambda self, ctx: (
            bpy.ops.gsplat.warn_scene_camera('INVOKE_DEFAULT')
            if self.use_scene_camera else None
        ),
    )
    show_all_cameras: BoolProperty(
        name="Show All Camera Positions",
        description="Draw every camera position and facing direction as a viewport overlay",
        default=False,
        update=lambda self, ctx: (
            _register_camera_overlay() if self.show_all_cameras
            else _unregister_camera_overlay()
        ),
    )
    export_lichtfeld: BoolProperty(
        name="Export to LichtFeld Studio",
        description="Automatically launch LichtFeld Studio and start training once the COLMAP export is complete",
        default=False,
    )
    lichtfeld_path: StringProperty(
        name="LichtFeld Executable",
        description="Full path to LichtFeld-Studio.exe",
        default="", subtype='FILE_PATH',
    )
    lichtfeld_strategy: EnumProperty(
        name="Strategy",
        description="Gaussian Splatting optimisation strategy",
        items=[
            ('mcmc',     "MCMC", "Markov Chain Monte Carlo — stable, uniform Gaussian distribution"),
            ('adc',      "ADC",  "Adaptive Density Control — the original 3DGS method"),
            ('igs_plus', "IGS+", "Improved Gaussian Splatting — enhanced densification heuristics"),
        ],
        default='mcmc',
    )
    lichtfeld_iterations: IntProperty(
        name="Iterations",
        description="Number of training iterations",
        default=30000, min=1000, max=1000000,
    )
    lichtfeld_max_gaussians: IntProperty(
        name="Max Gaussians",
        description="Maximum number of Gaussians (used by MCMC and IGS+)",
        default=1000000, min=1000, max=10000000,
    )
    export_masks: BoolProperty(
        name="Export Masks",
        description="Write a greyscale mask PNG per render (white = subject, black = background) "
                    "into masks/. Filename format: 00000.png.png",
        default=True,
    )


# ---------------------------------------------------------------------------
# Timer — render loop (main thread, no threading)
# ---------------------------------------------------------------------------

def _render_next_frame():
    if not _state["active"]:
        return None

    scene = bpy.context.scene
    props = scene.gsplat_props

    if _state["stop"]:
        _cleanup()
        props.progress = 0.0
        return None

    i       = _state["frame_index"]
    n_total = _state["total_frames"]

    if i >= n_total:
        if _state["colmap_frames"]:
            if not _state["colmap_pending"]:
                # Tick 1: show the message and yield so Blender can repaint.
                _state["colmap_pending"] = True
                props.status_message = "Generating point cloud…"
                props.progress = 100.0
                _tag_viewport_redraw()
                return 0.05
            # Tick 2: UI has repainted — now do the blocking work.
            output_dir = bpy.path.abspath(props.output_path)
            write_colmap_model(
                output_dir,
                _state["colmap_frames"],
                _state["camera_data"],
                scene.render,
                props.radius,
                props.point_cloud_points,
            )
            if props.export_lichtfeld and props.lichtfeld_path.strip():
                props.status_message = "Launching LichtFeld Studio…"
                _tag_viewport_redraw()
                _launch_lichtfeld(props, output_dir)
        _cleanup()
        return None

    if props.use_scene_camera:
        # Advance to the i-th frame and use the existing scene camera
        scene.frame_set(scene.frame_start + i)
        bpy.context.view_layer.update()
        pose = _blender_camera_to_colmap(scene.camera)
        lens = scene.camera.data.lens
        _state["colmap_frames"].append((i, *pose, lens))
    else:
        # Position the temporary camera at the i-th Fibonacci sphere point
        x, y, z = _state["positions"][i]
        cam = _state["camera_obj"]
        cam.location.x = x
        cam.location.y = y
        cam.location.z = z
        bpy.context.view_layer.update()
        pose = _blender_camera_to_colmap(cam)
        lens = _state["camera_data"].lens
        _state["colmap_frames"].append((i, *pose, lens))

    output_dir = bpy.path.abspath(props.output_path)
    images_dir = os.path.join(output_dir, "images")
    scene.render.filepath = os.path.join(images_dir, f"{i:05d}")

    bpy.ops.render.render(write_still=True)

    if props.transparent_background and props.export_masks:
        masks_dir = os.path.join(output_dir, "masks")
        _write_mask(scene.render.filepath, scene.render, masks_dir)

    _state["frame_index"] = i + 1
    props.progress = ((i + 1) / n_total) * 100.0
    return 0.001


# ---------------------------------------------------------------------------
# Render operator
# ---------------------------------------------------------------------------

class GSPLAT_OT_render(Operator):
    bl_idname      = "gsplat.render"
    bl_label       = "Render Gaussian Splat"
    bl_description = "Distribute cameras across a sphere and render all frames"
    bl_options     = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        props = context.scene.gsplat_props

        if _state["active"]:
            self.report({'WARNING'}, "A render is already running.")
            return {'CANCELLED'}

        output_dir = bpy.path.abspath(props.output_path)
        if not output_dir:
            self.report({'ERROR'}, "Output path is empty.")
            return {'CANCELLED'}

        if props.export_lichtfeld and not props.lichtfeld_path.strip():
            bpy.ops.gsplat.warn_lichtfeld_path('INVOKE_DEFAULT')
            return {'CANCELLED'}

        # Save scene state
        _state["original_camera"]           = context.scene.camera
        _state["original_output"]           = context.scene.render.filepath
        _state["original_frame"]            = context.scene.frame_current
        _state["original_world"]            = None
        _state["original_film_transparent"] = context.scene.render.film_transparent
        _state["original_color_mode"]       = context.scene.render.image_settings.color_mode

        if props.show_all_cameras:
            props.show_all_cameras = False
            _unregister_camera_overlay()

        _state["frame_index"]   = 0
        _state["colmap_frames"] = []
        _state["stop"]          = False
        _state["active"]        = True

        if props.transparent_background:
            context.scene.render.film_transparent = True
            context.scene.render.image_settings.color_mode = "RGBA"

        # Create output dirs
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        if props.transparent_background and props.export_masks:
            os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

        if props.use_scene_camera:
            # Render the existing scene camera across the scene's frame range
            if context.scene.camera is None:
                self.report({'ERROR'}, "No active scene camera found.")
                _state["active"] = False
                return {'CANCELLED'}
            n_frames = context.scene.frame_end - context.scene.frame_start + 1
            _state["total_frames"]  = n_frames
            _state["positions"]     = None
            _state["camera_obj"]    = None
            _state["camera_data"]   = context.scene.camera.data
            _state["target_obj"]    = None
        else:
            # Create temporary Fibonacci sphere cameras
            _state["total_frames"] = props.camera_count
            _state["positions"]    = fibonacci_sphere_positions(
                props.camera_count, props.radius,
                props.top_exclusion_angle, props.bottom_exclusion_angle,
                props.sphere_rot_x, props.sphere_rot_y, props.sphere_rot_z,
            )

            bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
            target      = context.active_object
            target.name = "_GSplat_Target"
            _state["target_obj"] = target

            cam_data = bpy.data.cameras.new(name="_GSplat_Camera")
            cam_data.lens = props.lens
            _state["original_lens"] = cam_data.lens

            cam_obj  = bpy.data.objects.new("_GSplat_Camera", cam_data)
            context.scene.collection.objects.link(cam_obj)
            con = cam_obj.constraints.new(type='TRACK_TO')
            con.target, con.track_axis, con.up_axis = target, 'TRACK_NEGATIVE_Z', 'UP_Y'
            context.scene.camera = cam_obj

            _state["camera_obj"]  = cam_obj
            _state["camera_data"] = cam_data

        props.is_rendering = True
        props.progress     = 0.0

        if not bpy.app.timers.is_registered(_render_next_frame):
            bpy.app.timers.register(_render_next_frame, first_interval=0.01)

        return {'FINISHED'}

    def execute(self, context):
        return self.invoke(context, None)


# ---------------------------------------------------------------------------
# Stop operator
# ---------------------------------------------------------------------------

class GSPLAT_OT_stop(Operator):
    bl_idname      = "gsplat.stop"
    bl_label       = "Stop Render"
    bl_description = "Stop after the current frame finishes"

    def execute(self, context):
        _state["stop"] = True
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Scene-camera warning popup
# ---------------------------------------------------------------------------

class GSPLAT_OT_warn_scene_camera(Operator):
    bl_idname   = "gsplat.warn_scene_camera"
    bl_label    = "Use Scene Camera — Warning"
    bl_options  = {'INTERNAL'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=460)

    def draw(self, context):
        layout = self.layout
        col    = layout.column(align=True)
        col.label(text="WARNING: Using the scene camera is not recommended.", icon='ERROR')
        col.separator()
        col.label(text="If there is any animation of objects in the scene (aside from")
        col.label(text="the camera itself) then the gaussian splat will not generate")
        col.label(text="correctly. All objects must be stationary for a gaussian splat")
        col.label(text="to interpret the data.")
        col.separator()
        col.label(text="It is recommended to use the addon's cameras in most cases.")

    def execute(self, context):
        # User clicked OK — keep use_scene_camera = True
        return {'FINISHED'}

    def cancel(self, context):
        # User dismissed — revert the toggle
        context.scene.gsplat_props.use_scene_camera = False


# ---------------------------------------------------------------------------
# LichtFeld path-missing warning popup
# ---------------------------------------------------------------------------

class GSPLAT_OT_warn_lichtfeld_path(Operator):
    bl_idname  = "gsplat.warn_lichtfeld_path"
    bl_label   = "LichtFeld Path Required"
    bl_options = {'INTERNAL'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=420)

    def draw(self, context):
        col = self.layout.column(align=True)
        col.label(text="LichtFeld Studio path is not set.", icon='ERROR')
        col.separator()
        col.label(text="Please provide the path to LichtFeld-Studio.exe")
        col.label(text="in the LichtFeld Studio section of the GSplat panel,")
        col.label(text="then try rendering again.")

    def execute(self, context):
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Preview operators
# ---------------------------------------------------------------------------

_preview_state = {
    "camera_obj": None, "camera_data": None,
    "target_obj": None, "original_camera": None,
}


class GSPLAT_OT_preview(Operator):
    bl_idname      = "gsplat.preview"
    bl_label       = "Preview Camera"
    bl_description = "Place a camera at the selected Fibonacci sphere position and look through it"
    bl_options     = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.gsplat_props
        idx   = max(0, min(props.preview_index, props.camera_count - 1))
        props.preview_index = idx

        positions = fibonacci_sphere_positions(
            props.camera_count, props.radius,
            props.top_exclusion_angle, props.bottom_exclusion_angle,
            props.sphere_rot_x, props.sphere_rot_y, props.sphere_rot_z,
        )
        idx = max(0, min(idx, len(positions) - 1))
        x, y, z = positions[idx]

        self._clear_preview(context, restore_camera=False)

        bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
        target      = context.active_object
        target.name = "_GSplat_Preview_Target"

        cam_data = bpy.data.cameras.new(name="_GSplat_Preview_Camera")
        cam_data.lens = props.lens
        cam_obj  = bpy.data.objects.new("_GSplat_Preview_Camera", cam_data)
        context.scene.collection.objects.link(cam_obj)
        cam_obj.location = (x, y, z)

        con = cam_obj.constraints.new(type='TRACK_TO')
        con.target, con.track_axis, con.up_axis = target, 'TRACK_NEGATIVE_Z', 'UP_Y'

        _preview_state.update({
            "camera_obj": cam_obj, "camera_data": cam_data,
            "target_obj": target,  "original_camera": context.scene.camera,
        })
        context.scene.camera = cam_obj

        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.region_3d.view_perspective = 'CAMERA'
                        break
                break

        props.is_previewing = True

        # Compute approximate spherical coordinates for the info message
        lat = math.degrees(math.asin(max(-1.0, min(1.0, z / props.radius))))
        lon = math.degrees(math.atan2(y, x)) % 360
        self.report({'INFO'}, f"Camera {idx}: lat={lat:.1f}°  lon={lon:.1f}°")
        return {'FINISHED'}

    def _clear_preview(self, context, restore_camera=True):
        if restore_camera and _preview_state["original_camera"]:
            context.scene.camera = _preview_state["original_camera"]
        for key in ("camera_obj", "target_obj"):
            obj = _preview_state.get(key)
            if obj and obj.name in bpy.data.objects:
                bpy.data.objects.remove(obj, do_unlink=True)
            _preview_state[key] = None
        cam_data = _preview_state.get("camera_data")
        if cam_data and cam_data.name in bpy.data.cameras:
            bpy.data.cameras.remove(cam_data)
        _preview_state["camera_data"] = _preview_state["original_camera"] = None
        context.scene.gsplat_props.is_previewing = False


class GSPLAT_OT_clear_preview(Operator):
    bl_idname      = "gsplat.clear_preview"
    bl_label       = "Clear Preview"
    bl_description = "Remove the preview camera and restore the original scene camera"
    bl_options     = {'REGISTER', 'UNDO'}

    def execute(self, context):
        orig = _preview_state.get("original_camera")
        if orig:
            context.scene.camera = orig
        for key in ("camera_obj", "target_obj"):
            obj = _preview_state.get(key)
            if obj and obj.name in bpy.data.objects:
                bpy.data.objects.remove(obj, do_unlink=True)
            _preview_state[key] = None
        cam_data = _preview_state.get("camera_data")
        if cam_data and cam_data.name in bpy.data.cameras:
            bpy.data.cameras.remove(cam_data)
        _preview_state["camera_data"] = _preview_state["original_camera"] = None
        context.scene.gsplat_props.is_previewing = False
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        if space.region_3d.view_perspective == 'CAMERA':
                            space.region_3d.view_perspective = 'PERSP'
                        break
                break
        self.report({'INFO'}, "Preview cleared.")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

class GSPLAT_PT_panel(Panel):
    bl_label       = "Gaussian Splat Generator"
    bl_idname      = "GSPLAT_PT_panel"
    bl_space_type  = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category    = "GSplat"

    def draw(self, context):
        layout = self.layout
        props  = context.scene.gsplat_props
        layout.use_property_split    = True
        layout.use_property_decorate = False

        if props.is_rendering:
            if not props.status_message:
                layout.label(text="Rendering in progress…", icon='RENDER_ANIMATION')
            layout.prop(props, "progress", slider=True)
            if props.status_message:
                layout.label(text=props.status_message, icon='SORTTIME')
            layout.separator()
            if not props.status_message:
                row = layout.row()
                row.scale_y = 1.6
                row.alert   = True
                row.operator("gsplat.stop", icon='CANCEL')
                layout.label(text="Press Esc to cancel too.", icon='INFO')

        else:
            # Camera mode selector
            layout.prop(props, "use_scene_camera", toggle=True,
                        icon='SCENE' if props.use_scene_camera else 'CAMERA_DATA')

            # Fibonacci camera distribution settings (disabled in scene-camera mode)
            layout.separator()
            col = layout.column(align=True)
            col.enabled = not props.use_scene_camera
            col.prop(props, "camera_count")
            col.prop(props, "radius")
            col.prop(props, "lens")
            col.separator()
            col.prop(props, "top_exclusion_angle",    slider=True)
            col.prop(props, "bottom_exclusion_angle", slider=True)
            col.separator()
            col.prop(props, "sphere_rot_x", slider=True)
            col.prop(props, "sphere_rot_y", slider=True)
            col.prop(props, "sphere_rot_z", slider=True)

            layout.separator()
            layout.prop(props, "output_path")

            # Preview (fibonacci cameras only)
            layout.separator()
            box = layout.box()
            box.enabled = not props.use_scene_camera

            # Show All Cameras — full-width toggle button matching "Preview in Viewport" style
            row_all = box.row()
            row_all.scale_y = 1.4
            icon_all = 'HIDE_OFF' if props.show_all_cameras else 'HIDE_ON'
            row_all.prop(props, "show_all_cameras", toggle=True, icon=icon_all)

            # Collapsible individual camera preview
            row_hdr = box.row()
            row_hdr.prop(
                props, "preview_expanded",
                text="Preview Individual Camera",
                icon='TRIA_DOWN' if props.preview_expanded else 'TRIA_RIGHT',
                emboss=False,
            )

            if props.preview_expanded:
                clamped = max(0, min(props.preview_index, props.camera_count - 1))
                box.prop(props, "preview_index", text="Camera Index")

                positions = fibonacci_sphere_positions(
                    props.camera_count, props.radius,
                    props.top_exclusion_angle, props.bottom_exclusion_angle,
                    props.sphere_rot_x, props.sphere_rot_y, props.sphere_rot_z,
                )
                clamped = max(0, min(clamped, len(positions) - 1))
                x, y, z = positions[clamped] if positions else (0.0, 0.0, props.radius)
                lat = math.degrees(math.asin(max(-1.0, min(1.0, z / max(props.radius, 0.001)))))
                lon = math.degrees(math.atan2(y, x)) % 360
                box.label(text=f"Camera {clamped}: lat={lat:.1f}°  lon={lon:.1f}°")

                row2 = box.row(align=True)
                row2.scale_y = 1.4
                if props.is_previewing:
                    row2.operator("gsplat.preview",       text="Update Preview", icon='FILE_REFRESH')
                    row2.operator("gsplat.clear_preview", text="Clear",          icon='X')
                else:
                    row2.operator("gsplat.preview", text="Preview in Viewport", icon='HIDE_OFF')

            # Export options
            layout.separator()
            layout.prop(props, "point_cloud_points")
            layout.prop(props, "transparent_background")
            if props.transparent_background:
                layout.prop(props, "export_masks")

            # LichtFeld Studio
            layout.separator()
            lf_box = layout.box()
            lf_box.prop(props, "export_lichtfeld")
            if props.export_lichtfeld:
                lf_col = lf_box.column(align=True)
                path_row = lf_col.row(align=True)
                path_row.alert = not props.lichtfeld_path.strip()
                path_row.prop(props, "lichtfeld_path", text="Path to Executable")
                lf_col.separator()
                lf_col.prop(props, "lichtfeld_strategy")
                lf_col.prop(props, "lichtfeld_iterations")
                lf_col.prop(props, "lichtfeld_max_gaussians")
                if props.transparent_background:
                    lf_col.separator()
                    lf_col.label(text="Mask: alpha_consistent (auto)", icon='INFO')

            layout.separator()
            row = layout.row()
            row.scale_y = 1.6
            row.operator("gsplat.render", icon='RENDER_ANIMATION')


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

classes = (
    GSplatProperties,
    GSPLAT_OT_render,
    GSPLAT_OT_stop,
    GSPLAT_OT_warn_scene_camera,
    GSPLAT_OT_warn_lichtfeld_path,
    GSPLAT_OT_preview,
    GSPLAT_OT_clear_preview,
    GSPLAT_PT_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.gsplat_props = bpy.props.PointerProperty(type=GSplatProperties)


def unregister():
    _unregister_camera_overlay()
    if _state["active"]:
        _state["stop"] = True
    if bpy.app.timers.is_registered(_render_next_frame):
        bpy.app.timers.unregister(_render_next_frame)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.gsplat_props


if __name__ == "__main__":
    register()