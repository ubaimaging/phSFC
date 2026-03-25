import os
import sys

import fcsparser
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.widgets import PolygonSelector
from phasorpy.phasor import phasor_from_signal
from phasorpy.plot import PhasorPlot

try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QColor
    from PyQt5.QtWidgets import (QApplication, QColorDialog, QComboBox,
                                 QFileDialog, QHBoxLayout, QLabel, QMainWindow,
                                 QMessageBox, QPushButton, QScrollArea,
                                 QSizePolicy, QVBoxLayout, QWidget)
except ImportError:
    try:
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QColor
        from PyQt6.QtWidgets import (QApplication, QColorDialog, QComboBox,
                                     QFileDialog, QHBoxLayout, QLabel,
                                     QMainWindow, QMessageBox, QPushButton,
                                     QScrollArea, QSizePolicy, QVBoxLayout,
                                     QWidget)
    except ImportError:
        raise ImportError("Please install PyQt5 or PyQt6: pip install PyQt5")


# Default color palette for successive gates
GATE_COLORS = [
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#fabed4",
    "#469990",
]


class PhasorGatingWindow(QMainWindow):
    """Qt-based interactive phasor gating tool supporting multiple simultaneous gates."""

    def __init__(
        self, fcs_path, g_data=None, s_data=None, spectral_columns=None, harmonic=1
    ):
        super().__init__()

        self.meta, self.data = fcsparser.parse(fcs_path, reformat_meta=True)
        self.filename = os.path.basename(fcs_path)
        self.harmonic = harmonic

        self.spectral_columns = self._determine_spectral_columns(spectral_columns)
        self.columns = list(self.data.columns)
        self.excluded_columns = [
            "Time",
            "SSC-H",
            "SSC-A",
            "FSC-H",
            "FSC-A",
            "SSC-B-H",
            "SSC-B-A",
        ]
        self.non_spectral_columns = [
            c for c in self.columns if c not in self.spectral_columns
        ]

        if g_data is not None and s_data is not None:
            self.G = np.asarray(g_data)
            self.S = np.asarray(s_data)
        else:
            self.G, self.S = self._compute_phasor()

        # Gates: list of dicts — {id, name, color, vertices, mask}
        self.gates = []
        self._next_gate_id = 1
        self._active_selector = None
        self._pending_color = None
        self._drawing = False

        default_cols = (
            self.non_spectral_columns if self.non_spectral_columns else self.columns
        )
        self.x_channel = default_cols[0]
        self.y_channel = default_cols[1] if len(default_cols) > 1 else default_cols[0]

        self.initUI()

    # ── Data helpers ──────────────────────────────────────────────────

    def _determine_spectral_columns(self, spectral_columns=None):
        all_cols = list(self.data.columns)
        if spectral_columns is not None:
            if isinstance(spectral_columns[0], int):
                return [all_cols[i] for i in spectral_columns]
            return list(spectral_columns)
        v_cols = [c for c in all_cols if "V" in c]
        if v_cols:
            return sorted(v_cols, key=lambda x: int(x.split("-")[0][1:]))
        return all_cols

    def _compute_phasor(self):
        intensity = self.data[self.spectral_columns].values
        _, real, imag = phasor_from_signal(intensity, harmonic=self.harmonic)
        return np.asarray(real).ravel(), np.asarray(imag).ravel()

    # ── UI ────────────────────────────────────────────────────────────

    def initUI(self):
        self.setWindowTitle(f"Phasor Gating Tool – {self.filename}")
        self.setGeometry(100, 100, 1700, 820)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(6)

        # ── Phasor canvas (left) ──────────────────────────────────────
        self.fig_phasor = Figure(figsize=(6, 6))
        self.canvas_phasor = FigureCanvas(self.fig_phasor)
        self.canvas_phasor.setMinimumWidth(480)
        self.canvas_phasor.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax_phasor = self.fig_phasor.add_subplot(111)

        phasor_toolbar = NavigationToolbar(self.canvas_phasor, self)
        phasor_widget = QWidget()
        phasor_vbox = QVBoxLayout(phasor_widget)
        phasor_vbox.setContentsMargins(0, 0, 0, 0)
        phasor_vbox.setSpacing(0)
        phasor_vbox.addWidget(phasor_toolbar)
        phasor_vbox.addWidget(self.canvas_phasor)
        main_layout.addWidget(phasor_widget, stretch=2)

        # ── Dot plots scroll area (middle) ────────────────────────────
        self.dot_scroll = QScrollArea()
        self.dot_scroll.setWidgetResizable(True)
        self.dot_container = QWidget()
        self.dot_hlayout = QHBoxLayout(self.dot_container)
        self.dot_hlayout.setAlignment(Qt.AlignLeft)
        self.dot_hlayout.setSpacing(8)
        self.dot_scroll.setWidget(self.dot_container)
        main_layout.addWidget(self.dot_scroll, stretch=3)

        # ── Control panel (right) ────────────────────────────────────
        ctrl = QWidget()
        ctrl.setFixedWidth(235)
        ctrl_layout = QVBoxLayout(ctrl)
        ctrl_layout.setAlignment(Qt.AlignTop)
        ctrl_layout.setSpacing(4)

        ctrl_layout.addWidget(QLabel("<b>X Axis:</b>"))
        self.combo_x = QComboBox()
        self.combo_x.addItems(self.columns)
        self.combo_x.setCurrentText(self.x_channel)
        self.combo_x.currentTextChanged.connect(self.on_x_changed)
        ctrl_layout.addWidget(self.combo_x)

        ctrl_layout.addSpacing(8)
        ctrl_layout.addWidget(QLabel("<b>Y Axis:</b>"))
        self.combo_y = QComboBox()
        self.combo_y.addItems(self.columns)
        self.combo_y.setCurrentText(self.y_channel)
        self.combo_y.currentTextChanged.connect(self.on_y_changed)
        ctrl_layout.addWidget(self.combo_y)

        ctrl_layout.addSpacing(14)
        self.btn_add = QPushButton("＋  Add Gate")
        self.btn_add.setStyleSheet(
            "QPushButton{background:#2196F3;color:white;font-weight:bold;"
            "padding:6px;border-radius:4px;}"
            "QPushButton:disabled{background:#aaa;}"
        )
        self.btn_add.clicked.connect(self.on_add_gate)
        ctrl_layout.addWidget(self.btn_add)

        self.status_lbl = QLabel("")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setStyleSheet("color:#e65100;font-style:italic;font-size:10px;")
        ctrl_layout.addWidget(self.status_lbl)

        ctrl_layout.addSpacing(10)
        ctrl_layout.addWidget(QLabel("<b>Gates:</b>"))

        # Scrollable gate list
        self.gate_list_scroll = QScrollArea()
        self.gate_list_scroll.setWidgetResizable(True)
        self.gate_list_scroll.setMaximumHeight(360)
        self.gate_list_widget = QWidget()
        self.gate_list_vbox = QVBoxLayout(self.gate_list_widget)
        self.gate_list_vbox.setAlignment(Qt.AlignTop)
        self.gate_list_vbox.setSpacing(3)
        self.gate_list_scroll.setWidget(self.gate_list_widget)
        ctrl_layout.addWidget(self.gate_list_scroll)

        ctrl_layout.addSpacing(14)
        btn_export = QPushButton("Export Plots (600 DPI)")
        btn_export.setStyleSheet(
            "QPushButton{background:#4CAF50;color:white;font-weight:bold;"
            "padding:6px;border-radius:4px;}"
        )
        btn_export.clicked.connect(self.on_export)
        ctrl_layout.addWidget(btn_export)
        ctrl_layout.addStretch()

        main_layout.addWidget(ctrl, stretch=0)

        # Initial render
        self._draw_phasor()
        self._rebuild_dot_plots()

        print("────────────────────────────────────────────────────────")
        print(f"  Spectral columns: {', '.join(self.spectral_columns[:5])}...")
        print("  Click '＋ Add Gate', draw a polygon on the phasor,")
        print("  then close it (click first vertex or press Enter).")
        print("  Repeat for as many gates as needed.")
        print("────────────────────────────────────────────────────────")

    # ── Phasor rendering ──────────────────────────────────────────────

    def _draw_phasor(self):
        """Redraw phasor background and all committed gate polygons."""
        # Safely disconnect any in-progress selector before clearing axes
        if self._active_selector is not None:
            try:
                self._active_selector.disconnect_events()
            except Exception:
                pass
            self._active_selector = None
            self._drawing = False
            self.btn_add.setEnabled(True)
            self.status_lbl.setText("")

        self.ax_phasor.clear()
        pp = PhasorPlot(
            allquadrants=True,
            title=f"{self.filename} – Harmonic {self.harmonic}",
            ax=self.ax_phasor,
        )
        pp.hist2d(self.G, self.S, bins=300, cmap="grey_r")

        for gate in self.gates:
            if gate["vertices"]:
                self.ax_phasor.add_patch(
                    mpatches.Polygon(
                        gate["vertices"],
                        closed=True,
                        edgecolor=gate["color"],
                        facecolor="none",
                        linewidth=4,
                        label=gate["name"],
                    )
                )

        self.canvas_phasor.draw_idle()

    # ── Dot plots rendering ───────────────────────────────────────────

    def _rebuild_dot_plots(self):
        """Rebuild scrollable area: one dot-plot widget per gate."""
        while self.dot_hlayout.count():
            item = self.dot_hlayout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        if not self.gates:
            lbl = QLabel(
                "No gates yet.\nUse '＋ Add Gate'\nand draw on the\nphasor plot."
            )
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("color:#888;font-size:13px;")
            self.dot_hlayout.addWidget(lbl)
            return

        x_all = self.data[self.x_channel].values
        y_all = self.data[self.y_channel].values
        n_total = len(x_all)

        for gate in self.gates:
            mask = gate["mask"]
            color = gate["color"]
            n_gated = int(mask.sum())
            pct = 100.0 * n_gated / n_total if n_total else 0.0

            fig = Figure(figsize=(4.5, 4.5))
            canvas = FigureCanvas(fig)
            canvas.setFixedSize(380, 380)
            ax = fig.add_subplot(111)

            ax.scatter(
                x_all[~mask],
                y_all[~mask],
                s=3,
                c="#aaaaaa",
                alpha=0.35,
                rasterized=True,
            )
            if n_gated > 0:
                ax.scatter(
                    x_all[mask],
                    y_all[mask],
                    s=6,
                    c=color,
                    alpha=0.75,
                    rasterized=True,
                    label=f"{n_gated} ({pct:.1f}%)",
                )
                ax.legend(loc="upper right", fontsize=8, markerscale=3)

            ax.set_xlabel(self.x_channel, fontsize=9)
            ax.set_ylabel(self.y_channel, fontsize=9)
            ax.set_title(
                f"{gate['name']}  –  {n_gated}/{n_total} ({pct:.1f}%)",
                fontsize=9,
                color=color,
                fontweight="bold",
            )
            if self.x_channel not in self.excluded_columns:
                ax.set_xscale("log")
            if self.y_channel not in self.excluded_columns:
                ax.set_yscale("log")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            fig.tight_layout()

            # Wrap canvas in a colored-border frame
            frame = QWidget()
            frame.setStyleSheet(
                f"background:white;border:2px solid {color};border-radius:4px;"
            )
            fl = QVBoxLayout(frame)
            fl.setContentsMargins(3, 3, 3, 3)
            fl.addWidget(canvas)
            frame.setFixedSize(390, 390)

            self.dot_hlayout.addWidget(frame)

    # ── Gate management ───────────────────────────────────────────────

    def on_add_gate(self):
        """Activate a new polygon selector for drawing a gate."""
        if self._drawing:
            return

        color = GATE_COLORS[len(self.gates) % len(GATE_COLORS)]
        self._pending_color = color
        self._drawing = True
        self.btn_add.setEnabled(False)
        self.status_lbl.setText(
            f"Drawing in {color}...\nClick vertices, close polygon by\n"
            "clicking first vertex or pressing Enter."
        )

        self._active_selector = PolygonSelector(
            self.ax_phasor,
            self._on_polygon_select,
            useblit=True,
            props=dict(color=color, linewidth=4),
            handle_props=dict(markersize=5),
        )

    def _on_polygon_select(self, verts):
        """Callback when the user completes a polygon."""
        path = Path(verts)
        points = np.column_stack([self.G, self.S])
        mask = path.contains_points(points)

        gate = {
            "id": self._next_gate_id,
            "name": f"Gate {self._next_gate_id}",
            "color": self._pending_color,
            "vertices": list(verts),
            "mask": mask,
        }
        self.gates.append(gate)
        self._next_gate_id += 1

        # Deactivate selector
        if self._active_selector is not None:
            try:
                self._active_selector.disconnect_events()
            except Exception:
                pass
            self._active_selector = None

        self._drawing = False
        self._pending_color = None
        self.btn_add.setEnabled(True)
        self.status_lbl.setText("")

        self._draw_phasor()
        self._rebuild_dot_plots()
        self._rebuild_gate_list()

    def _rebuild_gate_list(self):
        """Rebuild the gate-row list in the control panel."""
        while self.gate_list_vbox.count():
            item = self.gate_list_vbox.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        n_total = len(self.G)
        for gate in self.gates:
            n_gated = int(gate["mask"].sum())
            pct = 100.0 * n_gated / n_total if n_total else 0.0

            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(2, 2, 2, 2)
            rl.setSpacing(4)

            # Color swatch — click to change color
            btn_color = QPushButton()
            btn_color.setFixedSize(22, 22)
            btn_color.setStyleSheet(
                f"QPushButton{{background:{gate['color']};"
                f"border:1px solid #444;border-radius:3px;}}"
            )
            btn_color.setToolTip("Click to change gate color")
            btn_color.clicked.connect(lambda _, g=gate: self._pick_color(g))
            rl.addWidget(btn_color)

            # Name + count
            lbl = QLabel(f"<b>{gate['name']}</b>  {n_gated} ({pct:.1f}%)")
            lbl.setStyleSheet(
                f"color:{gate['color']};font-size:11px;"
                f"border:none;background:transparent;"
            )
            rl.addWidget(lbl, stretch=1)

            # Delete
            btn_del = QPushButton("✕")
            btn_del.setFixedSize(22, 22)
            btn_del.setStyleSheet(
                "QPushButton{color:#c00;font-weight:bold;"
                "border:none;background:transparent;}"
                "QPushButton:hover{background:#fee;}"
            )
            btn_del.setToolTip("Delete gate")
            btn_del.clicked.connect(lambda _, g=gate: self._delete_gate(g))
            rl.addWidget(btn_del)

            self.gate_list_vbox.addWidget(row)

    def _pick_color(self, gate):
        """Open a color dialog and update the gate color."""
        new_color = QColorDialog.getColor(
            QColor(gate["color"]), self, f"Color for {gate['name']}"
        )
        if new_color.isValid():
            gate["color"] = new_color.name()
            self._draw_phasor()
            self._rebuild_dot_plots()
            self._rebuild_gate_list()

    def _delete_gate(self, gate):
        """Remove a gate by id and renumber remaining gates sequentially."""
        self.gates = [g for g in self.gates if g["id"] != gate["id"]]
        for i, g in enumerate(self.gates, start=1):
            g["id"] = i
            g["name"] = f"Gate {i}"
        self._next_gate_id = len(self.gates) + 1
        self._draw_phasor()
        self._rebuild_dot_plots()
        self._rebuild_gate_list()

    # ── Axis callbacks ────────────────────────────────────────────────

    def on_x_changed(self, text):
        self.x_channel = text
        self._rebuild_dot_plots()

    def on_y_changed(self, text):
        self.y_channel = text
        self._rebuild_dot_plots()

    # ── Export ────────────────────────────────────────────────────────

    def on_export(self):
        """Export: phasor with all gates + one dot plot per gate (600 DPI)."""
        try:
            show_dirs = QFileDialog.ShowDirsOnly
        except AttributeError:
            show_dirs = QFileDialog.Option.ShowDirsOnly

        save_dir = QFileDialog.getExistingDirectory(
            self, "Select Directory to Save Plots", os.getcwd(), show_dirs
        )
        if not save_dir:
            return

        try:
            base = os.path.splitext(self.filename)[0]
            saved = []

            # ── Phasor with all gates overlaid ──
            fig_p = Figure(figsize=(8, 8))
            ax_p = fig_p.add_subplot(111)
            pp = PhasorPlot(allquadrants=True, title="", ax=ax_p)
            pp.hist2d(self.G, self.S, bins=300, cmap="gray_r")
            for gate in self.gates:
                if gate["vertices"]:
                    ax_p.add_patch(
                        mpatches.Polygon(
                            gate["vertices"],
                            closed=True,
                            edgecolor=gate["color"],
                            facecolor="none",
                            linewidth=4,
                            label=gate["name"],
                        )
                    )
            ax_p.tick_params(axis="both", which="major", labelsize=17)
            ax_p.set_xlabel("G, real", fontsize=19)
            ax_p.set_ylabel("S, imag", fontsize=19)
            fig_p.tight_layout()
            fname_phasor = os.path.join(save_dir, f"{base}_phasor_all_gates.png")
            fig_p.savefig(fname_phasor, dpi=600, bbox_inches="tight")
            plt.close(fig_p)
            saved.append(os.path.basename(fname_phasor))

            # ── One dot plot per gate ──
            x_all = self.data[self.x_channel].values
            y_all = self.data[self.y_channel].values
            n_total = len(x_all)

            for gate in self.gates:
                mask = gate["mask"]
                color = gate["color"]
                n_gated = int(mask.sum())
                pct = 100.0 * n_gated / n_total if n_total else 0.0

                fig_d = Figure(figsize=(8, 8))
                ax_d = fig_d.add_subplot(111)
                ax_d.scatter(
                    x_all[~mask],
                    y_all[~mask],
                    s=5,
                    c="#cccccc",
                    alpha=0.5,
                    rasterized=True,
                    label="Ungated",
                )
                if n_gated > 0:
                    ax_d.scatter(
                        x_all[mask],
                        y_all[mask],
                        s=10,
                        c=color,
                        alpha=0.8,
                        rasterized=True,
                        label=f"{gate['name']} · {n_gated} ({pct:.1f}%)",
                    )
                ax_d.tick_params(axis="both", which="major", labelsize=24)
                ax_d.set_xlabel(self.x_channel, fontsize=28)
                ax_d.set_ylabel(self.y_channel, fontsize=28)
                ax_d.set_ylim(1, 4e6)
                ax_d.set_xlim(1, 4e6)
                if self.x_channel not in self.excluded_columns:
                    ax_d.set_xscale("log")
                if self.y_channel not in self.excluded_columns:
                    ax_d.set_yscale("log")

                ax_d.spines["top"].set_visible(False)
                ax_d.spines["right"].set_visible(False)

                fig_d.tight_layout()

                safe = gate["name"].replace(" ", "_")
                fname_dot = os.path.join(
                    save_dir,
                    f"{base}_dotplot_{safe}_{self.x_channel}_vs_{self.y_channel}.png",
                )
                fig_d.savefig(fname_dot, dpi=600, bbox_inches="tight")
                plt.close(fig_d)
                saved.append(os.path.basename(fname_dot))

            QMessageBox.information(
                self,
                "Export Successful",
                f"Saved {len(saved)} file(s) to:\n{save_dir}\n\n" + "\n".join(saved),
            )
            print(f"✓ Exported {len(saved)} file(s) to {save_dir}")
            for f in saved:
                print(f"  - {f}")

        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Error:\n{str(e)}")
            print(f"✗ Export failed: {e}")

    # ── Public API ────────────────────────────────────────────────────

    def get_masks(self):
        """Return dict of gate_name → mask array."""
        return {g["name"]: g["mask"] for g in self.gates}

    def get_mask(self):
        """Return union mask of all gates (backward compatibility)."""
        if not self.gates:
            return np.zeros(len(self.G), dtype=bool)
        return np.any([g["mask"] for g in self.gates], axis=0)


# ====================================================================== #
#  Convenience wrapper (backward compatibility)
# ====================================================================== #


class PhasorGatingTool:
    def __init__(
        self, fcs_path, g_data=None, s_data=None, spectral_columns=None, harmonic=1
    ):
        self.fcs_path = fcs_path
        self.g_data = g_data
        self.s_data = s_data
        self.spectral_columns = spectral_columns
        self.harmonic = harmonic
        self.window = None

    def run(self, gui_channel_picker=True, figsize=(14, 6)):
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        self.window = PhasorGatingWindow(
            self.fcs_path,
            self.g_data,
            self.s_data,
            self.spectral_columns,
            self.harmonic,
        )
        self.window.show()

        try:
            app.exec()
        except AttributeError:
            app.exec_()

        return self.window.get_mask()


# ====================================================================== #
#  Entry point
# ====================================================================== #

if __name__ == "__main__":
    if len(sys.argv) < 2:
        fcs_files = [f for f in os.listdir(".") if f.lower().endswith(".fcs")]
        if not fcs_files:
            print("Usage: python interactive_BAL_pregated.py <file.fcs>")
            sys.exit(1)
        fcs_path = fcs_files[0]
    else:
        fcs_path = sys.argv[1]

    print(f"Loading {fcs_path} ...")
    tool = PhasorGatingTool(fcs_path, harmonic=1)
    mask = tool.run()
    print(
        f"\nDone. {mask.sum()} events selected ({100 * mask.sum() / len(mask):.1f}%)."
    )
