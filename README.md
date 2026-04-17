# Gaussian Splat Generator — Blender Add On

A Blender add on that automatically distributes cameras across a sphere, renders them, and exports a COLMAP-compatible dataset ready for 3D Gaussian Splatting training.

---

## Requirements

- **Blender 5.1.0** or later
- **LichtFeld Studio** (for training the Gaussian Splat) — [Download here](https://lichtfeld.io/)

---

## Installation
**Option 1: **
1. Download the `.zip` file in this repository
2. Click the dropdown in the top-right corner and choose **Install from Disk**
3. Select the generated `.zip` file

**Option 2:**
1. Download or clone this repository
2. Build the extension `.zip` from the repository root:
   ```
   blender --command extension build
   ```
3. In Blender, go to **Edit → Preferences → Get Extensions**
4. Click the dropdown in the top-right corner and choose **Install from Disk**
5. Select the generated `.zip` file

---

## Usage

1. Open your Blender scene with the object(s) you want to capture
2. In the **3D Viewport**, open the sidebar (`N` key) and go to the **GSplat** tab, the add on will position cameras in a sphere sourrounding the origin with paramaters that you can set. Additionally, the scene camera can be used to create the splat. If this is used, ensure that all objects in the scene are static as movement will cause the splat to generate incorrectly. 
3. Configure your settings:
   - **Camera Count** — number of cameras distributed across the sphere
   - **Radius** — distance of cameras from the origin
   - **Focal Length** — camera lens in mm
   - **Top / Bottom Exclusion** — exclude cameras near the poles
   - **Sphere Rotation** — rotate the entire camera rig
   - **Output Path** — where renders and COLMAP data will be saved
4. Optionally use **Show All Camera Positions** to preview the camera rig in the viewport
5. Set the **Path to Executable** under the LichtFeld Studio section to your `LichtFeld-Studio.exe`
6. Click **Render Gaussian Splat** to begin

The addon will:
- Render all camera positions to `<output>/images/`
- Optionally export alpha masks to `<output>/masks/`
- Write a COLMAP sparse model to `<output>/sparse/0/`
- Automatically launch LichtFeld Studio for training (if configured)

---

## LichtFeld Studio

LichtFeld Studio is the recommended tool for training 3D Gaussian Splats from the exported COLMAP data as it is free and open source. Alternatively, this add-on generates a COLMAP database that can be used in any other gausian splat generation tool. 

**Download:** [https://lichtfeld.io/](https://lichtfeld.io/)

Once installed, set the path to `LichtFeld-Studio.exe` in the addon panel. The addon will launch it automatically after rendering completes.

### Training Strategies

| Strategy | Description |
|----------|-------------|
| **MCMC** | Markov Chain Monte Carlo — stable, uniform Gaussian distribution |
| **ADC**  | Adaptive Density Control — the original 3DGS method |
| **IGS+** | Improved Gaussian Splatting — enhanced densification heuristics |

---

## Output Structure

```
<output_path>/
├─ images/          # Rendered frames (one per camera position)
├─ masks/           # Alpha masks (if Transparent Background is enabled)
└─ sparse/
   └─ 0/
      ├─ cameras.txt
      ├─ images.txt
      └─ points3D.txt
```

---

## License

[GNU General Public License v3.0](LICENSE)
