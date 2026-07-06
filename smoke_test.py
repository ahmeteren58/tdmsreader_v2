"""Headless smoke test for tdmsreader changes.

Covers:
1. _read_channel_strided correctness on an open-mode nptdms channel
   (including the lazy overview / view paths).
2. Worker-level loading of analog + digital + large (lazy) channels.
3. PlotPane routing: auto (digital -> right), manual left/right overrides,
   all-right X range handling, and y_shift on digital right bounds.
"""

import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from nptdms import TdmsWriter, ChannelObject, TdmsFile

import tdmsreader_release_ready_no_stats as app_mod

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


# ------------------------------------------------------------------
# Build a synthetic TDMS file
# ------------------------------------------------------------------
tmpdir = tempfile.mkdtemp(prefix="tdms_smoke_")
path = os.path.join(tmpdir, "synthetic.tdms")

n_small = 10_000
n_big = 2_500_000  # above lazy fullload_threshold (2M)
t = np.arange(n_small) / 1000.0
analog1 = np.sin(2 * np.pi * 5 * t).astype(np.float64)
analog2 = (1000.0 + 500.0 * np.cos(2 * np.pi * 2 * t)).astype(np.float64)  # different scale
digital = ((np.arange(n_small) // 500) % 2).astype(np.float64)
big = np.sin(np.arange(n_big) * 0.001).astype(np.float64)

props = {"wf_increment": 0.001, "wf_start_offset": 0.0}
with TdmsWriter(path) as w:
    w.write_segment([
        ChannelObject("G1", "analog1", analog1, properties=props),
        ChannelObject("G1", "analog2", analog2, properties=props),
        ChannelObject("G1", "digital1", digital, properties=props),
    ])
    # write big channel in segments
    seg = 500_000
    for i in range(0, n_big, seg):
        w.write_segment([ChannelObject("G2", "big", big[i:i + seg], properties=props)])

print(f"TDMS written: {path} ({os.path.getsize(path)/1e6:.1f} MB)")

# ------------------------------------------------------------------
# 1. _read_channel_strided correctness on open-mode channel
# ------------------------------------------------------------------
with TdmsFile.open(path) as tdms:
    ch = tdms["G2"]["big"]
    for (i0, i1, stride) in [(0, n_big, 13), (100, 5000, 1), (999, 1_200_345, 997), (0, 10, 3)]:
        got = app_mod._read_channel_strided(ch, i0, i1, stride, chunk_samples=300_000)
        expected = big[i0:i1:stride]
        check(
            f"strided read ({i0}:{i1}:{stride})",
            got.shape == expected.shape and np.allclose(got, expected),
            f"shape {got.shape} vs {expected.shape}",
        )

    # strided slice on open-mode file must not be supported directly
    # (this is why the chunked fallback matters)
    try:
        _ = ch[0:1000:7]
        direct_ok = True
    except Exception:
        direct_ok = False
    print(f"[INFO] nptdms open-mode supports strided slicing directly: {direct_ok}")

# ------------------------------------------------------------------
# 2. Worker-level load (analog + digital + lazy big channel)
# ------------------------------------------------------------------
from PyQt6.QtWidgets import QApplication

qapp = QApplication(sys.argv)

files = {"F1": {"path": path, "label": "synthetic.tdms"}}
reqs = [
    app_mod.ChannelRequest(app_mod.ChannelKey("F1", "G1", "analog1", n_small), "analog1"),
    app_mod.ChannelRequest(app_mod.ChannelKey("F1", "G1", "analog2", n_small), "analog2"),
    app_mod.ChannelRequest(app_mod.ChannelKey("F1", "G1", "digital1", n_small), "digital1"),
    app_mod.ChannelRequest(app_mod.ChannelKey("F1", "G2", "big", n_big), "big"),
]

result = {}
worker = app_mod.MultiTdmsChannelLoadWorker(files, reqs)
worker.finished.connect(lambda p: result.update(p))
worker.failed.connect(lambda m: result.update({"error": m}))
worker.run()

check("worker load no error", "error" not in result, result.get("error", ""))
series = result.get("series", [])
check("worker loaded 4 series", len(series) == 4, f"got {len(series)}")

by_name = {s["name"]: s for s in series}
check("digital detected", bool(by_name.get("digital1", {}).get("is_digital")))
check("analog1 not digital", not by_name.get("analog1", {}).get("is_digital", True))
check("big channel lazy", bool(by_name.get("big", {}).get("lazy")))
check(
    "lazy overview limited points",
    0 < len(by_name.get("big", {}).get("y", [])) <= app_mod.CFG.lazy.overview_max_points + 1,
    f"{len(by_name.get('big', {}).get('y', []))} pts",
)

# ------------------------------------------------------------------
# 3. LazyViewLoadWorker (visible window reads)
# ------------------------------------------------------------------
lazy_req = [{
    "style_key": by_name["big"]["style_key"],
    "path": path, "group": "G2", "channel": "big", "n": n_big,
    "x_mode": "seconds", "x_base": 0.0, "x_inc": 0.001,
    "x0": 100.0, "x1": 110.0,
}]
lv_result = {}
lv = app_mod.LazyViewLoadWorker(lazy_req, max_points=250_000)
lv.finished.connect(lambda p: lv_result.update(p))
lv.failed.connect(lambda m: lv_result.update({"error": m}))
lv.run()
upd = lv_result.get("updates", {}).get(by_name["big"]["style_key"])
check("lazy view returned window", upd is not None and len(upd["y"]) > 0)
if upd is not None:
    i0, i1, stride = upd["win"]
    expected = big[i0:i1:stride]
    check("lazy view data matches file", np.allclose(np.asarray(upd["y"]), expected))

# ------------------------------------------------------------------
# 4. PlotPane routing (auto + manual axis overrides)
# ------------------------------------------------------------------
pane = app_mod.PlotPane(bg_rgb=(255, 255, 255))
small_series = [by_name["analog1"], by_name["analog2"], by_name["digital1"]]

# 4a. auto: digital -> right, both analogs -> left
pane.plot_series(small_series, "numeric", "Zaman (s)", style_map={})
check("auto: 2 left items", len(pane._left_items) == 2, f"{len(pane._left_items)}")
check("auto: 1 right item (digital)", len(pane._right_items) == 1, f"{len(pane._right_items)}")

# 4b. Excel-style: assign analog2 to the right axis
smap = {by_name["analog2"]["style_key"]: {"axis": "right"}}
pane.plot_series(small_series, "numeric", "Zaman (s)", style_map=smap)
check("manual Y2: 1 left item", len(pane._left_items) == 1, f"{len(pane._left_items)}")
check("manual Y2: 2 right items", len(pane._right_items) == 2, f"{len(pane._right_items)}")
rb = pane._right_data_bounds
check(
    "right bounds include analog2 scale",
    rb is not None and rb[0] <= 500.0 and rb[1] >= 1500.0,
    f"bounds={rb}",
)

# 4c. force digital to the left axis
smap2 = {by_name["digital1"]["style_key"]: {"axis": "left"}}
pane.plot_series(small_series, "numeric", "Zaman (s)", style_map=smap2)
check("digital->left: 3 left items", len(pane._left_items) == 3, f"{len(pane._left_items)}")
check("digital->left: 0 right items", len(pane._right_items) == 0, f"{len(pane._right_items)}")
check("digital->left: right axis destroyed", pane._right_vb is None)

# 4d. everything on the right: X range must still be set from right data
smap3 = {s["style_key"]: {"axis": "right"} for s in small_series}
pane.plot_series(small_series, "numeric", "Zaman (s)", style_map=smap3)
check("all-right: 0 left items", len(pane._left_items) == 0)
check("all-right: 3 right items", len(pane._right_items) == 3)
xr = pane.plot.getViewBox().viewRange()[0]
check(
    "all-right: X range covers data",
    xr[0] <= 0.5 and xr[1] >= 9.0,
    f"xrange={xr}",
)

# 4e. y_shift applied to digital right bounds
smap4 = {by_name["digital1"]["style_key"]: {"y_shift": 10.0}}
pane.plot_series(small_series, "numeric", "Zaman (s)", style_map=smap4)
rb = pane._right_data_bounds
check(
    "digital y_shift shifts right bounds",
    rb is not None and abs(rb[0] - 10.0) < 1e-9 and abs(rb[1] - 11.0) < 1e-9,
    f"bounds={rb}",
)

# ------------------------------------------------------------------
# 5. MainWindow end-to-end (axis combobox behaviour)
# ------------------------------------------------------------------
win = app_mod.MainWindow()
win.current_series = [dict(s) for s in small_series]
win._refresh_series_comboboxes()

# select analog2 in the style combo
idx = next(
    i for i in range(win.cmb_style_series.count())
    if win.cmb_style_series.itemData(i) == by_name["analog2"]["style_key"]
)
win.cmb_style_series.setCurrentIndex(idx)
check("axis combo defaults to auto", win._current_axis_side() == "auto")

# choose "Sağ Eksen (Y2)" -> should update style_map immediately
right_idx = next(
    i for i in range(win.cmb_axis_side.count())
    if win.cmb_axis_side.itemData(i) == "right"
)
win.cmb_axis_side.setCurrentIndex(right_idx)
st = win.style_map.get(by_name["analog2"]["style_key"], {})
check("combo change writes style_map axis", st.get("axis") == "right", f"style={st}")

# switching back to auto removes the key
auto_idx = next(
    i for i in range(win.cmb_axis_side.count())
    if win.cmb_axis_side.itemData(i) == "auto"
)
win.cmb_axis_side.setCurrentIndex(auto_idx)
st = win.style_map.get(by_name["analog2"]["style_key"], {})
check("auto removes axis override", "axis" not in st, f"style={st}")

# reset restores combo
win.cmb_axis_side.setCurrentIndex(right_idx)
win.reset_style_for_selected_series()
check("reset returns combo to auto", win._current_axis_side() == "auto")

# ------------------------------------------------------------------
# 6. Axis renaming (overrides on pane + MainWindow sync)
# ------------------------------------------------------------------
def axis_label_text(p, axis):
    return str(p.plot.getPlotItem().getAxis(axis).labelText or "")

# pane-level override survives replot and applies to labels
pane.set_axis_label_override("left", "Basınç (bar)")
pane.set_axis_label_override("bottom", "Süre")
pane.plot_series(small_series, "numeric", "Zaman (s)", style_map={})
check("left axis override applied", axis_label_text(pane, "left") == "Basınç (bar)",
      axis_label_text(pane, "left"))
check("bottom axis override applied", axis_label_text(pane, "bottom") == "Süre",
      axis_label_text(pane, "bottom"))

# right override applies while a right axis exists (digital1 -> auto Y2)
pane.set_axis_label_override("right", "Vana Durumu")
check("right axis override applied", axis_label_text(pane, "right") == "Vana Durumu",
      axis_label_text(pane, "right"))

# clearing returns to automatic label
pane.set_axis_label_override("left", "")
check("left axis reset to auto", axis_label_text(pane, "left") != "Basınç (bar)",
      axis_label_text(pane, "left"))
check("effective label reported", pane.effective_axis_label("bottom") == "Süre")

# axis hit-test finds axes at their scene positions
pane.resize(800, 600)
pane.show()
qapp.processEvents()
p1 = pane.plot.getPlotItem()
left_ax = p1.getAxis("left")
bottom_ax = p1.getAxis("bottom")
left_center = left_ax.mapRectToScene(left_ax.rect()).center()
bottom_center = bottom_ax.mapRectToScene(bottom_ax.rect()).center()
check("hit-test left axis", pane._axis_at_scene_pos(left_center) == "left",
      str(pane._axis_at_scene_pos(left_center)))
check("hit-test bottom axis", pane._axis_at_scene_pos(bottom_center) == "bottom",
      str(pane._axis_at_scene_pos(bottom_center)))
vb_center = pane.plot.getViewBox().sceneBoundingRect().center()
check("hit-test plot area is not an axis", pane._axis_at_scene_pos(vb_center) is None)

# MainWindow: double-click rename signal path syncs fields + panes
win._on_axis_label_edited("left", "Tork (Nm)")
check("mainwindow stores override", win.axis_label_overrides.get("left") == "Tork (Nm)")
check("field synced", win.ed_axis_y1.text() == "Tork (Nm)")
check("main pane label synced",
      str(win.plot_pane.plot.getPlotItem().getAxis("left").labelText or "") == "Tork (Nm)")

# 'Eksenler' tab apply/reset
win.ed_axis_x.setText("Deney Süresi (s)")
win._apply_axis_labels_from_fields()
check("controls-tab apply works", win.axis_label_overrides.get("bottom") == "Deney Süresi (s)")
win._reset_axis_labels()
check("controls-tab reset clears overrides", win.axis_label_overrides == {})
check("fields cleared after reset", win.ed_axis_x.text() == "" and win.ed_axis_y1.text() == "")

win.close()

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURES: {FAILURES}")
    sys.exit(1)
print("ALL TESTS PASSED")
