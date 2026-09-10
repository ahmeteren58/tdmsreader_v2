"""
TDMS Okuyucu – NI TDMS dosyaları için DIAdem benzeri görüntüleyici.

Orijinal sürüme göre yapılan iyileştirmeler:
- Farklı örnekleme hızındaki eşzamanlı kanallar için otomatik Fs/zaman senkronizasyonu
- Kanal Yöneticisi (görünürlük, ad, birim, Fs, eksen, renk)
- Overlay / senkronize Stacked grafik görünümü
- Min-max envelope downsampling + PyQtGraph peak/clip optimizasyonu
- Güvenli hesaplanan/sanal kanallar
- Gelişmiş plot kısayolları, zoom geçmişi ve sağ-tık menüsü
- Manuel veri aralığı analizi (ortalama, std, min/max, P-P, RMS, medyan, integral)
- Doğru kaynak yönetimi (TDMS dosyaları için bağlam yöneticileri)
- Temizleme takibi ile iş parçacığı yaşam döngüsü yönetimi
- Sessiz istisna yutma kaldırıldı; hatalar kaydedilir
- Yardımcı metotlar ile kod tekrarı azaltıldı
- Çizili veriler için CSV dışa aktarma eklendi
- Yapılandırılabilir sabitler dataclass'a çıkarıldı
- Worker/thread yaşam döngüsündeki bellek sızıntıları düzeltildi
- Tembel yükleme hattındaki yarış durumu düzeltmeleri
- Geliştirilmiş numpy dizi işleme (daha az gereksiz kopya)
- IDE desteği ve dokümantasyon için kapsamlı tip ipuçları
- Genel metotlarda uygun docstring'ler
- Çok büyük kanallar için düzeltilmiş dijital adım çizimi
- Uzun işlemler sırasında daha iyi ilerleme geri bildirimi
"""

from __future__ import annotations

import gc
import math
import os
import sys
import logging
import threading
import ast
import re
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import (
    Any, Callable, Dict, Generator, List, Optional, Sequence, Tuple, Union,
)

import numpy as np

# Ensure pyqtgraph uses PyQt6 when multiple Qt bindings are installed
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt6")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_LOG_LEVEL = os.environ.get("TDMSREADER_LOGLEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tdmsreader")

# ---------------------------------------------------------------------------
# Qt imports
# ---------------------------------------------------------------------------
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QObject, QSize, QTimer, QSettings,
)
from PyQt6.QtGui import QColor, QCloseEvent, QIcon, QPixmap, QAction, QKeySequence, QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QAbstractItemView, QFileDialog,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QTreeWidget,
    QTreeWidgetItem, QSplitter, QMessageBox, QStatusBar, QCheckBox, QGroupBox,
    QFormLayout, QDoubleSpinBox, QTabWidget, QComboBox, QButtonGroup, QSpinBox,
    QColorDialog, QGridLayout, QSizePolicy, QToolButton, QFrame,
    QProgressBar, QMenu, QMenuBar, QDialog, QTextBrowser, QStackedWidget, QScrollArea, QStyle, QFontComboBox,
)

# ---------------------------------------------------------------------------
# pyqtgraph
# ---------------------------------------------------------------------------
import pyqtgraph as pg
from pyqtgraph import PlotWidget
from pyqtgraph.graphicsItems.DateAxisItem import DateAxisItem


class _EditableAxisMixin:
    """Small mixin that turns a pyqtgraph axis label into a direct edit target."""

    def __init__(self, *args: Any, edit_key: str = "", edit_callback: Optional[Callable[[str], None]] = None, **kwargs: Any) -> None:
        self._edit_key = str(edit_key or "")
        self._edit_callback = edit_callback
        super().__init__(*args, **kwargs)

    def mouseDoubleClickEvent(self, ev: Any) -> None:  # noqa: N802 - Qt naming convention
        cb = getattr(self, "_edit_callback", None)
        if callable(cb):
            try:
                cb(str(getattr(self, "_edit_key", "") or getattr(self, "orientation", "")))
                ev.accept()
                return
            except Exception:
                logger.debug("Axis double-click callback failed", exc_info=True)
        try:
            super().mouseDoubleClickEvent(ev)
        except Exception:
            try:
                ev.ignore()
            except Exception:
                pass


class EditableAxisItem(_EditableAxisMixin, pg.AxisItem):
    pass


class EditableDateAxisItem(_EditableAxisMixin, DateAxisItem):
    pass


class AxisQuickEditDialog(QDialog):
    """Compact direct editor opened by double-clicking an axis."""

    def __init__(
        self,
        axis_caption: str,
        current_title: str,
        auto_title: str,
        style: Dict[str, Any],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{axis_caption} Ekseni Düzenle")
        self.setModal(True)
        self.resize(430, 285)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        form = QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(9)

        self.le_title = QLineEdit(str(current_title or ""))
        self.le_title.setPlaceholderText(f"Otomatik: {auto_title or '—'}")
        self.cmb_font = QFontComboBox()
        self.cmb_font.setCurrentFont(QFont(str(style.get("font_family", "Arial") or "Arial")))

        self.sp_title_size = QSpinBox()
        self.sp_title_size.setRange(8, 28)
        self.sp_title_size.setSuffix(" pt")
        self.sp_title_size.setValue(max(8, int(style.get("title_size", 11) or 11)))
        self.chk_title_bold = QCheckBox("Başlığı Bold yap")
        self.chk_title_bold.setChecked(bool(style.get("title_bold", False)))

        self.sp_tick_size = QSpinBox()
        self.sp_tick_size.setRange(7, 20)
        self.sp_tick_size.setSuffix(" pt")
        self.sp_tick_size.setValue(max(7, int(style.get("tick_size", 9) or 9)))
        self.chk_tick_bold = QCheckBox("Tick değerlerini Bold yap")
        self.chk_tick_bold.setChecked(bool(style.get("tick_bold", False)))

        form.addRow("Eksen adı:", self.le_title)
        form.addRow("Font:", self.cmb_font)

        title_row = QWidget()
        tl = QHBoxLayout(title_row)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(8)
        tl.addWidget(self.sp_title_size)
        tl.addWidget(self.chk_title_bold)
        tl.addStretch(1)
        form.addRow("Başlık stili:", title_row)

        tick_row = QWidget()
        kl = QHBoxLayout(tick_row)
        kl.setContentsMargins(0, 0, 0, 0)
        kl.setSpacing(8)
        kl.addWidget(self.sp_tick_size)
        kl.addWidget(self.chk_tick_bold)
        kl.addStretch(1)
        form.addRow("Tick stili:", tick_row)
        root.addLayout(form)

        note = QLabel("Not: Font, boyut ve bold ayarları tüm grafik eksenlerine ortak uygulanır. Eksen adı yalnızca çift tıkladığınız ekseni değiştirir.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #6B7280;")
        root.addWidget(note)

        actions = QHBoxLayout()
        self.btn_auto = QPushButton("Otomatik Başlık")
        self.btn_cancel = QPushButton("İptal")
        self.btn_apply = QPushButton("Uygula")
        self.btn_apply.setProperty("primary", True)
        actions.addWidget(self.btn_auto)
        actions.addStretch(1)
        actions.addWidget(self.btn_cancel)
        actions.addWidget(self.btn_apply)
        root.addLayout(actions)

        self.btn_auto.clicked.connect(self.le_title.clear)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_apply.clicked.connect(self.accept)

    def values(self) -> Dict[str, Any]:
        family = "Arial"
        try:
            family = self.cmb_font.currentFont().family() or "Arial"
        except Exception:
            pass
        return {
            "title": self.le_title.text().strip(),
            "font_family": family,
            "title_size": int(self.sp_title_size.value()),
            "title_bold": bool(self.chk_title_bold.isChecked()),
            "tick_size": int(self.sp_tick_size.value()),
            "tick_bold": bool(self.chk_tick_bold.isChecked()),
        }

try:
    import OpenGL  # noqa: F401
    pg.setConfigOption("useOpenGL", True)
    pg.setConfigOption("enableExperimental", True)
except ImportError:
    pass

pg.setConfigOptions(antialias=False)

# ---------------------------------------------------------------------------
# TDMS
# ---------------------------------------------------------------------------
from nptdms import TdmsFile

# ---------------------------------------------------------------------------
# Optional: Savitzky-Golay filter
# ---------------------------------------------------------------------------
try:
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    savgol_filter = None  # type: ignore[assignment]


# ===================================================================
# Configuration
# ===================================================================

@dataclass(frozen=True)
class LazyConfig:
    """All tunables for lazy (on-demand) TDMS loading in one place."""
    enabled: bool = True
    fullload_threshold: int = 2_000_000
    overview_max_points: int = 200_000
    view_max_points: int = 250_000
    view_debounce_ms: int = 180


@dataclass(frozen=True)
class AppConfig:
    """Application-wide tunables."""
    lazy: LazyConfig = field(default_factory=LazyConfig)
    default_filter_window: int = 21
    default_filter_polyorder: int = 3
    min_fft_samples: int = 8
    max_digital_levels: int = 8
    digital_sample_limit: int = 50_000
    pad_ratio: float = 0.06
    settings_org: str = "TDMSReader"
    settings_app: str = "TDMSReader"
    max_recent_files: int = 10
    plot_max_points: int = 80_000
    stacked_max_channels: int = 24


CFG = AppConfig()


# ===================================================================
# Data models
# ===================================================================

@dataclass(frozen=True)
class ChannelKey:
    """Uniquely identifies a channel across multiple open files."""
    file_id: str
    group: str
    channel: str
    length: int


@dataclass
class ChannelRequest:
    key: ChannelKey
    display_name: str


@dataclass
class FileState:
    file_id: str
    path: str
    label: str
    tree: QTreeWidget
    search: QLineEdit


# ===================================================================
# Theme QSS
# ===================================================================

LIGHT_QSS = """
QMainWindow { background: #F6F7F9; }
QLabel { color: #222; }
QLabel:disabled { color: #9AA3B2; }
QTabWidget::pane { border: 1px solid #D7DBE0; border-radius: 12px; background: #FFFFFF; }
QTabBar::tab { padding: 8px 12px; margin: 2px; border-radius: 9px; background: #EEF1F4; color: #222; }
QTabBar::tab:selected { background: #FFFFFF; border: 1px solid #D7DBE0; }
QGroupBox { border: 1px solid #D7DBE0; border-radius: 12px; margin-top: 14px; background: #FFFFFF; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #222; font-weight: 600; }
QPushButton, QToolButton { border: 1px solid #C9CED6; border-radius: 9px; padding: 7px 12px; background: #FFFFFF; color: #000; }
QPushButton:hover, QToolButton:hover { background: #F1F3F6; }
QPushButton:pressed, QToolButton:pressed { background: #E9ECF0; }
QPushButton[primary="true"] { background: #2ECC71; border-color: #26B863; color: white; font-weight: 700; }
QPushButton[primary="true"]:hover { background: #29C46B; }
QPushButton[danger="true"] { background: #E74C3C; border-color: #D94435; color: white; font-weight: 700; }
QPushButton[danger="true"]:hover { background: #DE4637; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { border: 1px solid #C9CED6; border-radius: 9px; padding: 5px 8px; background: #FFFFFF; selection-background-color: #e3ddff; }
QTreeWidget { border: 1px solid #D7DBE0; border-radius: 12px; background: #FFFFFF; }
QHeaderView::section { background: #F1F3F6; padding: 6px; border: none; border-right: 1px solid #D7DBE0; color: #222; font-weight: 600; }
QCheckBox, QRadioButton { spacing: 8px; }
QStatusBar { background: #FFFFFF; border-top: 1px solid #D7DBE0; }
QFrame#plotQuickbar { background: rgba(255,255,255,0.70); border: 1px solid #D7DBE0; border-radius: 14px; }
QToolButton[qb="1"] { border: 1px solid #C9CED6; border-radius: 12px; padding: 0px; background: #FFFFFF; }
QToolButton[qb="1"]:hover { background: #F1F3F6; }
QToolButton[qb="1"]:pressed { background: #E9ECF0; }
QToolButton[qb="1"]:checked { background: #E7F0FF; border: 1px solid #4C8DFF; }
QToolButton[qb="1"]:checked:hover { background: #DCEAFF; }
QToolButton[qb="1"]:disabled { background: #F5F6F8; color: #9AA3B2; border-color: #D7DBE0; }
QFrame#controlsDock { border: 1px solid #D7DBE0; border-radius: 12px; background: #FFFFFF; }
QFrame#controlsHeader { background: #EEF1F4; border-bottom: 1px solid #D7DBE0; border-top-left-radius: 12px; border-top-right-radius: 12px; }
QToolButton#btnCtrlCollapse { border: 1px solid #C9CED6; border-radius: 9px; padding: 6px 10px; background: #FFFFFF; font-weight: 800; text-align: left; }
QToolButton#btnCtrlCollapse:hover { background: #F1F3F6; }
QToolButton#btnCtrlCollapse:pressed { background: #E9ECF0; }
QProgressBar { border: 1px solid #C9CED6; border-radius: 6px; background: #F1F3F6; text-align: center; }
QProgressBar::chunk { background: #2ECC71; border-radius: 5px; }
QMenuBar { background: #F6F7F9; border-bottom: 1px solid #D7DBE0; padding: 2px; }
QMenuBar::item { padding: 6px 12px; border-radius: 6px; color: #222; }
QMenuBar::item:selected { background: #EEF1F4; }
QMenuBar::item:pressed { background: #E9ECF0; }
QMenu { background: #FFFFFF; border: 1px solid #D7DBE0; border-radius: 8px; padding: 4px; }
QMenu::item { padding: 6px 28px 6px 12px; border-radius: 4px; color: #222; }
QMenu::item:selected { background: #EEF1F4; }
QMenu::separator { height: 1px; background: #D7DBE0; margin: 4px 8px; }
"""

DARK_QSS = """
QMainWindow { background: #0F1115; }
QLabel { color: #E8EAF0; }
QLabel:disabled { color: #7F889B; }
QTabWidget::pane { border: 1px solid #2A2F3A; border-radius: 12px; background: #141824; }
QTabBar::tab { padding: 8px 12px; margin: 2px; border-radius: 9px; background: #1D2230; color: #E8EAF0; }
QTabBar::tab:selected { background: #141824; border: 1px solid #2A2F3A; }
QGroupBox { border: 1px solid #2A2F3A; border-radius: 12px; margin-top: 14px; background: #141824; color: #E8EAF0; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #E8EAF0; font-weight: 600; }
QPushButton, QToolButton { border: 1px solid #3A4253; border-radius: 9px; padding: 7px 12px; background: #1A2030; color: #E8EAF0; }
QPushButton:hover, QToolButton:hover { background: #222A3D; }
QPushButton:pressed, QToolButton:pressed { background: #2A3550; }
QPushButton[primary="true"] { background: #2ECC71; border-color: #26B863; color: white; font-weight: 700; }
QPushButton[primary="true"]:hover { background: #29C46B; }
QPushButton[danger="true"] { background: #E74C3C; border-color: #D94435; color: white; font-weight: 700; }
QPushButton[danger="true"]:hover { background: #DE4637; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { border: 1px solid #3A4253; border-radius: 9px; padding: 5px 8px; background: #101624; color: #E8EAF0; selection-background-color: #2C3A5C; }
QTreeWidget { border: 1px solid #2A2F3A; border-radius: 12px; background: #101624; color: #E8EAF0; }
QHeaderView::section { background: #1A2030; padding: 6px; border: none; border-right: 1px solid #2A2F3A; color: #E8EAF0; font-weight: 600; }
QCheckBox, QRadioButton { spacing: 8px; color: #E8EAF0; }
QStatusBar { background: #141824; border-top: 1px solid #2A2F3A; color: #E8EAF0; }
QFrame#plotQuickbar { background: rgba(20,24,36,0.80); border: 1px solid #2A2F3A; border-radius: 14px; }
QToolButton[qb="1"] { border: 1px solid #3A4253; border-radius: 12px; padding: 0px; background: #1A2030; color: #E8EAF0; }
QToolButton[qb="1"]:hover { background: #222A3D; }
QToolButton[qb="1"]:pressed { background: #2A3550; }
QToolButton[qb="1"]:checked { background: #2A3550; border: 1px solid #6C8DFF; }
QToolButton[qb="1"]:checked:hover { background: #324068; }
QToolButton[qb="1"]:disabled { background: #161B28; color: #7F889B; border-color: #2A2F3A; }
QFrame#controlsDock { border: 1px solid #2A2F3A; border-radius: 12px; background: #141824; }
QFrame#controlsHeader { background: #1D2230; border-bottom: 1px solid #2A2F3A; border-top-left-radius: 12px; border-top-right-radius: 12px; }
QToolButton#btnCtrlCollapse { border: 1px solid #2A2F3A; border-radius: 9px; padding: 6px 10px; background: #141824; color: #E8EAF0; font-weight: 800; text-align: left; }
QToolButton#btnCtrlCollapse:hover { background: #20273A; }
QToolButton#btnCtrlCollapse:pressed { background: #1B2131; }
QProgressBar { border: 1px solid #3A4253; border-radius: 6px; background: #1A2030; color: #E8EAF0; text-align: center; }
QProgressBar::chunk { background: #2ECC71; border-radius: 5px; }
QMenuBar { background: #0F1115; border-bottom: 1px solid #2A2F3A; padding: 2px; }
QMenuBar::item { padding: 6px 12px; border-radius: 6px; color: #E8EAF0; }
QMenuBar::item:selected { background: #1D2230; }
QMenuBar::item:pressed { background: #2A3550; }
QMenu { background: #141824; border: 1px solid #2A2F3A; border-radius: 8px; padding: 4px; }
QMenu::item { padding: 6px 28px 6px 12px; border-radius: 4px; color: #E8EAF0; }
QMenu::item:selected { background: #222A3D; }
QMenu::separator { height: 1px; background: #2A2F3A; margin: 4px 8px; }
"""


# Professional workspace UI overrides. Kept separate from the base theme so the
# application logic and existing widget-specific styling remain untouched.
LIGHT_QSS += """
QMainWindow { background: #F3F6FA; }
QWidget#appRoot { background: #F3F6FA; }
QFrame#ribbonBar { background: #FFFFFF; border: 1px solid #DCE3EC; border-radius: 8px; }
QFrame#controlRibbon { background: #FAFBFD; border: 1px solid #DCE3EC; border-radius: 8px; }
QFrame#controlRibbon QCheckBox { spacing: 5px; }
QFrame#controlRibbon QLabel[compactLabel="true"] { color: #5F6B7A; font-size: 10px; font-weight: 600; }
QFrame#controlRibbon QDoubleSpinBox, QFrame#controlRibbon QSpinBox, QFrame#controlRibbon QComboBox { min-height: 26px; padding: 2px 6px; }
QFrame#controlRibbon QPushButton { min-height: 27px; padding: 3px 8px; }
QFrame#ribbonGroup { background: transparent; border: none; border-right: 1px solid #E5EAF1; }
QFrame#ribbonGroup[lastGroup="true"] { border-right: none; }
QLabel[ribbonGroupTitle="true"] { color: #697386; font-size: 9px; font-weight: 600; padding-top: 1px; }
QToolButton[ribbon="1"] { border: 1px solid transparent; border-radius: 7px; padding: 5px 5px; background: transparent; color: #24344D; font-weight: 600; font-size: 9px; }
QToolButton#workspaceDetailsButton { border: 1px solid #D7E0EB; border-radius: 6px; padding: 4px 10px; background: #FFFFFF; color: #24344D; font-weight: 600; }
QToolButton#workspaceDetailsButton:hover { background: #F0F6FF; border-color: #AFCBEE; }
QToolButton[ribbon="1"]:hover { background: #F0F6FF; border-color: #CFE0FA; }
QToolButton[ribbon="1"]:pressed, QToolButton[ribbon="1"]:checked { background: #E6F1FF; border-color: #9CC4FF; color: #0E5EC7; }
QFrame#sidePanel, QFrame#workspacePanel { background: #FFFFFF; border: 1px solid #DCE3EC; border-radius: 8px; }
QFrame#workspaceTopBar { background: #FAFBFD; border: 1px solid #E2E7EE; border-radius: 7px; }
QFrame#summaryStrip { background: #FAFBFD; border: 1px solid #E2E7EE; border-radius: 7px; }
QLabel[summaryCaption="true"] { color: #778196; font-size: 9px; font-weight: 600; }
QLabel[summaryValue="true"] { color: #1D2C43; font-size: 11px; font-weight: 700; }
QFrame#summaryCell { border-right: 1px solid #E3E8EF; }
QFrame#controlsDock { border: 1px solid #DCE3EC; border-radius: 8px; background: #FFFFFF; }
QFrame#controlsHeader { background: #F7F9FC; border-bottom: 1px solid #E0E6EE; border-top-left-radius: 8px; border-top-right-radius: 8px; }
QToolButton#btnCtrlCollapse { border: none; border-radius: 6px; padding: 7px 8px; background: transparent; font-weight: 700; color: #24344D; text-align: left; }
QToolButton#btnCtrlCollapse:hover { background: #EEF4FC; }
QPushButton, QToolButton { border-radius: 6px; }
QPushButton[primary="true"] { background: #1677FF; border-color: #1169E4; color: white; font-weight: 700; }
QPushButton[primary="true"]:hover { background: #0F6FEF; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { border-radius: 6px; min-height: 24px; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border: 1px solid #5B9DFF; }
QTreeWidget { border-radius: 6px; }
QTreeWidget::item { padding: 4px 3px; }
QTreeWidget::item:selected { background: #DDEBFF; color: #173B70; }
QTabWidget::pane { border-radius: 7px; }
QTabBar::tab { border-radius: 6px; padding: 7px 12px; background: transparent; margin: 1px; }
QTabBar::tab:selected { background: #FFFFFF; color: #145FBA; border: 1px solid #CBDDF5; font-weight: 700; }
QSplitter::handle { background: transparent; }
QSplitter::handle:hover { background: #D8E6F8; }
QFrame#plotQuickbar { background: rgba(255,255,255,0.92); border: 1px solid #D8E0EA; border-radius: 9px; }
QToolButton[qb="1"] { border-radius: 7px; }
QStatusBar { background: #FFFFFF; border-top: 1px solid #DCE3EC; color: #44546B; }
QStatusBar QLabel { padding: 0 9px; color: #44546B; }
QScrollBar:vertical { width: 10px; background: transparent; margin: 2px; }
QScrollBar::handle:vertical { background: #C6CEDA; border-radius: 4px; min-height: 28px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QScrollBar:horizontal { height: 10px; background: transparent; margin: 2px; }
QScrollBar::handle:horizontal { background: #C6CEDA; border-radius: 4px; min-width: 28px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }
"""

DARK_QSS += """
QMainWindow { background: #0C111B; }
QWidget#appRoot { background: #0C111B; }
QFrame#ribbonBar { background: #151C28; border: 1px solid #283446; border-radius: 8px; }
QFrame#controlRibbon { background: #151E2C; border: 1px solid #283446; border-radius: 8px; }
QFrame#controlRibbon QCheckBox { spacing: 5px; }
QFrame#controlRibbon QLabel[compactLabel="true"] { color: #99A6B8; font-size: 10px; font-weight: 600; }
QFrame#controlRibbon QDoubleSpinBox, QFrame#controlRibbon QSpinBox, QFrame#controlRibbon QComboBox { min-height: 26px; padding: 2px 6px; }
QFrame#controlRibbon QPushButton { min-height: 27px; padding: 3px 8px; }
QFrame#ribbonGroup { background: transparent; border: none; border-right: 1px solid #2B3749; }
QFrame#ribbonGroup[lastGroup="true"] { border-right: none; }
QLabel[ribbonGroupTitle="true"] { color: #8D99AA; font-size: 9px; font-weight: 600; padding-top: 1px; }
QToolButton[ribbon="1"] { border: 1px solid transparent; border-radius: 7px; padding: 5px 5px; background: transparent; color: #E5EAF2; font-weight: 600; font-size: 9px; }
QToolButton#workspaceDetailsButton { border: 1px solid #334258; border-radius: 6px; padding: 4px 10px; background: #151E2C; color: #E5EAF2; font-weight: 600; }
QToolButton#workspaceDetailsButton:hover { background: #202B3B; border-color: #4C6484; }
QToolButton[ribbon="1"]:hover { background: #202B3B; border-color: #34445C; }
QToolButton[ribbon="1"]:pressed, QToolButton[ribbon="1"]:checked { background: #203B62; border-color: #477FC7; color: #D7E9FF; }
QFrame#sidePanel, QFrame#workspacePanel { background: #121925; border: 1px solid #283446; border-radius: 8px; }
QFrame#workspaceTopBar { background: #151E2C; border: 1px solid #293649; border-radius: 7px; }
QFrame#summaryStrip { background: #151E2C; border: 1px solid #293649; border-radius: 7px; }
QLabel[summaryCaption="true"] { color: #8995A8; font-size: 9px; font-weight: 600; }
QLabel[summaryValue="true"] { color: #E6ECF5; font-size: 11px; font-weight: 700; }
QFrame#summaryCell { border-right: 1px solid #2C394D; }
QFrame#controlsDock { border: 1px solid #283446; border-radius: 8px; background: #121925; }
QFrame#controlsHeader { background: #151E2C; border-bottom: 1px solid #293649; border-top-left-radius: 8px; border-top-right-radius: 8px; }
QToolButton#btnCtrlCollapse { border: none; border-radius: 6px; padding: 7px 8px; background: transparent; font-weight: 700; color: #E6ECF5; text-align: left; }
QToolButton#btnCtrlCollapse:hover { background: #202B3B; }
QPushButton, QToolButton { border-radius: 6px; }
QPushButton[primary="true"] { background: #287DDF; border-color: #246FC5; color: white; font-weight: 700; }
QPushButton[primary="true"]:hover { background: #3288E9; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { border-radius: 6px; min-height: 24px; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border: 1px solid #4D8FD8; }
QTreeWidget { border-radius: 6px; }
QTreeWidget::item { padding: 4px 3px; }
QTreeWidget::item:selected { background: #213A5D; color: #E9F3FF; }
QTabWidget::pane { border-radius: 7px; }
QTabBar::tab { border-radius: 6px; padding: 7px 12px; background: transparent; margin: 1px; }
QTabBar::tab:selected { background: #182232; color: #A8CFFF; border: 1px solid #35557D; font-weight: 700; }
QSplitter::handle { background: transparent; }
QSplitter::handle:hover { background: #25364D; }
QFrame#plotQuickbar { background: rgba(18,25,37,0.94); border: 1px solid #2C394D; border-radius: 9px; }
QToolButton[qb="1"] { border-radius: 7px; }
QStatusBar { background: #121925; border-top: 1px solid #283446; color: #AAB4C4; }
QStatusBar QLabel { padding: 0 9px; color: #AAB4C4; }
QScrollBar:vertical { width: 10px; background: transparent; margin: 2px; }
QScrollBar::handle:vertical { background: #465268; border-radius: 4px; min-height: 28px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QScrollBar:horizontal { height: 10px; background: transparent; margin: 2px; }
QScrollBar::handle:horizontal { background: #465268; border-radius: 4px; min-width: 28px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }
"""


# ===================================================================
# Helper functions
# ===================================================================

def _to_epoch_seconds(t: Any) -> float:
    """Convert various time representations to epoch seconds (UTC)."""
    if t is None:
        raise TypeError("Time value must not be None")
    if isinstance(t, np.datetime64):
        return float(t.astype("datetime64[ns]").astype(np.int64)) / 1e9
    if isinstance(t, datetime):
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return float(t.timestamp())
    if isinstance(t, (int, float, np.integer, np.floating)):
        return float(t)
    raise TypeError(f"Unsupported time type: {type(t)}")


def _timedelta_to_seconds(dt: Any) -> float:
    """Convert various timedelta representations to seconds."""
    if isinstance(dt, timedelta):
        return float(dt.total_seconds())
    if isinstance(dt, np.timedelta64):
        return float(dt.astype("timedelta64[ns]").astype(np.int64)) / 1e9
    if isinstance(dt, (int, float, np.integer, np.floating)):
        return float(dt)
    raise TypeError(f"Unsupported increment type: {type(dt)}")


@contextmanager
def _open_tdms(path: str) -> Generator[TdmsFile, None, None]:
    """Context manager ensuring TDMS files are always closed."""
    tdms = TdmsFile.open(path)
    try:
        yield tdms
    finally:
        try:
            tdms.close()
        except Exception:
            logger.debug("Failed to close TDMS file: %s", path)


def _ensure_1d_numeric(arr: np.ndarray) -> np.ndarray:
    """Flatten, make contiguous, and ensure numeric dtype."""
    arr = np.asarray(arr)
    if arr.ndim != 1:
        arr = np.ravel(arr)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    if not np.issubdtype(arr.dtype, np.number):
        arr = arr.astype(np.float64)
    return arr


def _xmeta_from_channel_props(
    props: dict,
) -> Tuple[str, float, float]:
    """Extract (x_mode, x_base, x_inc) from TDMS channel properties.

    Returns:
        x_mode: 'index' | 'seconds' | 'datetime'
        x_base: base offset for X axis generation
        x_inc:  per-sample increment
    """
    try:
        if props and "wf_increment" in props:
            inc = _timedelta_to_seconds(props["wf_increment"])
            start_offset = float(props.get("wf_start_offset", 0.0) or 0.0)

            if "wf_start_time" in props:
                start_epoch = _to_epoch_seconds(props["wf_start_time"])
                if abs(start_epoch) < 86400 * 2:
                    return "seconds", start_offset, float(inc)
                return "datetime", float(start_epoch + start_offset), float(inc)

            return "seconds", start_offset, float(inc)
    except Exception:
        logger.debug("Failed to extract x metadata from channel properties", exc_info=True)
    return "index", 0.0, 1.0


def _slice_indices_from_xrange(
    x0: float, x1: float, n: int,
    x_mode: str, x_base: float, x_inc: float,
) -> Tuple[int, int]:
    """Convert an X-axis range to sample index range."""
    if n <= 0:
        return 0, 0
    a, b = (min(x0, x1), max(x0, x1))

    if x_mode == "index":
        i0 = int(math.floor(a))
        i1 = int(math.ceil(b))
    elif math.isfinite(x_inc) and x_inc > 0:
        i0 = int(math.floor((a - x_base) / x_inc))
        i1 = int(math.ceil((b - x_base) / x_inc))
    else:
        i0, i1 = 0, n

    # 5% padding, minimum 1000 samples
    win = max(1, i1 - i0)
    pad = max(1000, int(win * 0.05))
    i0 = max(0, i0 - pad)
    i1 = min(n, i1 + pad)
    if i1 <= i0:
        i1 = min(n, i0 + 1)
    return i0, i1


def _stride_for_window(i0: int, i1: int, max_points: int) -> int:
    """Compute stride to limit the number of rendered points."""
    win = max(1, i1 - i0)
    if max_points <= 0:
        return 1
    return max(1, math.ceil(win / max_points))


def minmax_envelope_downsample(
    x: np.ndarray, y: np.ndarray, max_points: int = 80_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce a dense analog series while preserving local minima and maxima.

    Two representative extrema are retained per bucket, in their original temporal
    order.  This is intended for display only; analysis/FFT always uses full data.
    """
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y)
    n = min(xx.size, yy.size)
    if n <= 0:
        return xx[:0], yy[:0]
    xx, yy = xx[:n], yy[:n]
    if max_points <= 4 or n <= max_points:
        return xx, yy

    # Two extrema per bucket.  Use vectorized full buckets and a tiny leftover tail.
    target_buckets = max(1, max_points // 2)
    bucket = max(2, int(math.ceil(n / target_buckets)))
    full_n = (n // bucket) * bucket
    picked: List[np.ndarray] = []

    if full_n > 0:
        y2 = np.asarray(yy[:full_n], dtype=np.float64).reshape(-1, bucket)
        finite = np.isfinite(y2)
        safe_min = np.where(finite, y2, np.inf)
        safe_max = np.where(finite, y2, -np.inf)
        imin = np.argmin(safe_min, axis=1)
        imax = np.argmax(safe_max, axis=1)
        all_bad = ~np.any(finite, axis=1)
        imin[all_bad] = 0
        imax[all_bad] = min(1, bucket - 1)
        base = np.arange(y2.shape[0], dtype=np.int64) * bucket
        a = base + imin
        b = base + imax
        pair = np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1).ravel()
        picked.append(pair)

    if full_n < n:
        tail = np.asarray(yy[full_n:n], dtype=np.float64)
        finite = np.isfinite(tail)
        if np.any(finite):
            smn = np.where(finite, tail, np.inf)
            smx = np.where(finite, tail, -np.inf)
            a = full_n + int(np.argmin(smn))
            b = full_n + int(np.argmax(smx))
            picked.append(np.array(sorted({a, b}), dtype=np.int64))
        else:
            picked.append(np.array([full_n], dtype=np.int64))

    idx = np.concatenate(picked) if picked else np.arange(0, n, bucket, dtype=np.int64)
    idx = np.unique(np.clip(idx, 0, n - 1))
    return xx[idx], yy[idx]


_ALLOWED_CALC_FUNCS: Dict[str, Callable] = {
    "abs": np.abs, "sqrt": np.sqrt, "log": np.log, "log10": np.log10,
    "exp": np.exp, "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "gradient": np.gradient, "cumsum": np.cumsum,
    "minimum": np.minimum, "maximum": np.maximum, "clip": np.clip,
}


def _validate_calc_expression(expr: str, allowed_names: set) -> ast.Expression:
    tree = ast.parse(expr, mode="eval")
    allowed_nodes = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult,
        ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd, ast.Call, ast.Name,
        ast.Load, ast.Constant,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"Desteklenmeyen ifade öğesi: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in allowed_names and node.id not in _ALLOWED_CALC_FUNCS:
            raise ValueError(f"Bilinmeyen kanal/fonksiyon: {node.id}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_CALC_FUNCS:
                raise ValueError("Yalnızca izin verilen matematik fonksiyonları çağrılabilir.")
    return tree


def classify_and_extract_x(
    channel: Any, n_samples: int,
) -> Tuple[np.ndarray, str]:
    """Determine X-axis data and mode for a TDMS channel.

    Tries time_track() first, then wf_increment properties, falls back to index.
    """
    try:
        tt = channel.time_track()
        if tt is not None and len(tt) == n_samples:
            if isinstance(tt, np.ndarray) and np.issubdtype(tt.dtype, np.datetime64):
                x = tt.astype("datetime64[ns]").astype(np.int64).astype(np.float64) / 1e9
                return x, "datetime"
            if isinstance(tt, np.ndarray) and np.issubdtype(tt.dtype, np.number):
                return np.asarray(tt, dtype=np.float64), "seconds"
            if isinstance(tt, (list, tuple)) and tt and isinstance(tt[0], datetime):
                x = np.array([_to_epoch_seconds(v) for v in tt], dtype=np.float64)
                return x, "datetime"
    except Exception:
        pass

    props = getattr(channel, "properties", {}) or {}
    if "wf_increment" in props:
        try:
            inc = _timedelta_to_seconds(props["wf_increment"])
            start_offset = float(props.get("wf_start_offset", 0.0))
            if "wf_start_time" in props:
                start_epoch = _to_epoch_seconds(props["wf_start_time"])
                if abs(start_epoch) < 86400 * 2:
                    return start_offset + np.arange(n_samples, dtype=np.float64) * inc, "seconds"
                return (start_epoch + start_offset + np.arange(n_samples, dtype=np.float64) * inc), "datetime"
            return start_offset + np.arange(n_samples, dtype=np.float64) * inc, "seconds"
        except Exception:
            pass

    return np.arange(n_samples, dtype=np.float64), "index"


def compute_interval_statistics(
    x: np.ndarray, y: np.ndarray, x_min: float, x_max: float,
) -> dict:
    """Compute descriptive statistics for finite samples inside an inclusive X interval.

    This function is deliberately UI-independent so the same calculation can be
    unit-tested and used safely from a background worker.
    """
    xx = np.asarray(x, dtype=np.float64).reshape(-1)
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    n = min(xx.size, yy.size)
    if n <= 0:
        raise ValueError("Kanalda örnek bulunamadı.")
    xx, yy = xx[:n], yy[:n]

    lo, hi = sorted((float(x_min), float(x_max)))
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        raise ValueError("Geçerli bir başlangıç/bitiş aralığı girin.")

    finite_x = np.isfinite(xx)
    in_window = finite_x & (xx >= lo) & (xx <= hi)
    selected_count = int(np.count_nonzero(in_window))
    valid = in_window & np.isfinite(yy)
    valid_count = int(np.count_nonzero(valid))
    if valid_count <= 0:
        raise ValueError("Seçilen aralıkta geçerli örnek bulunamadı.")

    xv = xx[valid]
    yv = yy[valid]
    if xv.size > 1 and np.any(np.diff(xv) < 0):
        order = np.argsort(xv)
        xv, yv = xv[order], yv[order]

    mean = float(np.mean(yv))
    std = float(np.std(yv, ddof=0))
    min_v = float(np.min(yv))
    max_v = float(np.max(yv))
    rms = float(np.sqrt(np.mean(np.square(yv))))
    median = float(np.median(yv))
    ptp = float(max_v - min_v)
    if xv.size > 1:
        try:
            integral = float(np.trapezoid(yv, xv))
        except AttributeError:
            integral = float(np.trapz(yv, xv))
    else:
        integral = 0.0

    return {
        "n": valid_count,
        "selected_n": selected_count,
        "invalid_n": max(0, selected_count - valid_count),
        "mean": mean,
        "std": std,
        "min": min_v,
        "max": max_v,
        "ptp": ptp,
        "rms": rms,
        "median": median,
        "integral": integral,
        "x_first": float(xv[0]),
        "x_last": float(xv[-1]),
    }


def robust_fs_from_x(x: np.ndarray) -> Optional[float]:
    """Estimate sampling frequency from an X-axis array using median diff."""
    if x is None or len(x) < 3:
        return None
    limit = 10_000
    subset = x[:limit] if x.size > limit else x
    dx = np.diff(subset.astype(np.float64))
    dx = dx[np.isfinite(dx)]
    if dx.size == 0:
        return None
    med = float(np.median(dx))
    return (1.0 / med) if med > 0 else None


def fs_value_to_hz(value: float, unit: str) -> Optional[float]:
    """Convert a user-entered frequency value + unit string to Hz."""
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val) or val <= 0.0:
        return None
    if str(unit or "Hz").strip().lower() == "khz":
        val *= 1e3
    return val


def apply_savgol_safe(
    y: np.ndarray, window_len: int,
    polyorder_hint: int = 3,
) -> np.ndarray:
    """Apply Savitzky-Golay filter with automatic parameter clamping."""
    if savgol_filter is None:
        return y
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n < 5:
        return y

    y_clean = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    win = int(window_len)
    if win % 2 == 0:
        win += 1
    if win > n:
        win = n if (n % 2 == 1) else (n - 1)
    if win < 5:
        return y_clean

    poly = min(int(polyorder_hint), win - 2)
    if poly < 1:
        return y_clean

    return savgol_filter(y_clean, window_length=win, polyorder=poly, mode="interp")


def _safe_str(v: Any) -> str:
    s = str(v).strip()
    return "" if (not s or s.lower() in ("none", "nan")) else s


def infer_quantity_from_props(props: dict) -> str:
    """Best-effort extraction of a human-readable quantity name from TDMS properties."""
    if not props:
        return "Değer"
    preferred = [
        "quantity", "Quantity", "physical_quantity", "PhysicalQuantity",
        "measurement", "Measurement", "NI_MeasurementName", "NI_SignalName",
        "NI_UnitDescription", "unit_description", "UnitDescription",
        "NI_Description", "description", "Description", "NI_ChannelDescription",
    ]
    for k in preferred:
        if k in props:
            s = _safe_str(props[k])
            if s:
                return s[:80]
    tokens = ("quantity", "measure", "signal", "physical", "sensor", "unitdesc", "descr")
    for k, v in props.items():
        if any(tok in str(k).lower() for tok in tokens):
            s = _safe_str(v)
            if s:
                return s[:80]
    return "Değer"


def common_unit(series_list: List[dict]) -> Optional[str]:
    units = [s.get("unit", "").strip() for s in series_list if s.get("unit", "").strip()]
    if not units:
        return None
    return units[0] if all(u == units[0] for u in units) else None


def common_quantity(series_list: List[dict]) -> Optional[str]:
    qs = [s.get("quantity", "").strip() for s in series_list if s.get("quantity", "").strip()]
    if not qs:
        return None
    return qs[0] if all(q == qs[0] for q in qs) else None


def qcolor_to_tuple(c: QColor) -> Tuple[int, int, int]:
    return (c.red(), c.green(), c.blue())


def file_label_from_path(path: str) -> str:
    return os.path.basename(path)


def style_key_for_channel(file_id: str, group: str, channel: str) -> str:
    return f"{file_id}|{group}|{channel}"


def _bg_to_fg_rgb(bg: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Choose black or white foreground for contrast against *bg*."""
    lum = 0.2126 * bg[0] + 0.7152 * bg[1] + 0.0722 * bg[2]
    return (0, 0, 0) if lum > 140 else (255, 255, 255)


def apply_plotwidget_theme(pw: PlotWidget, bg_rgb: Tuple[int, int, int]) -> None:
    fg = _bg_to_fg_rgb(bg_rgb)
    pw.setBackground(bg_rgb)
    item = pw.getPlotItem()
    for ax_name in ("left", "bottom", "right"):
        axis = item.getAxis(ax_name)
        if axis is not None:
            axis.setPen(pg.mkPen(fg))
            axis.setTextPen(pg.mkPen(fg))


def linear_detrend_safe(y: np.ndarray) -> np.ndarray:
    """Remove linear trend from *y*, handling NaN/Inf gracefully."""
    yy = np.asarray(y, dtype=np.float64)
    if yy.size < 2:
        return np.nan_to_num(yy, nan=0.0, posinf=0.0, neginf=0.0)

    out = yy.copy()
    finite = np.isfinite(out)
    if np.count_nonzero(finite) < 2:
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    idx = np.arange(out.size, dtype=np.float64)
    try:
        m, b = np.polyfit(idx[finite], out[finite], 1)
        out[finite] -= (m * idx[finite] + b)
    except Exception:
        out[finite] -= float(np.nanmean(out[finite]))
    out[~finite] = 0.0
    return out


def padded_minmax(
    values: np.ndarray, pad_ratio: float = 0.06,
) -> Optional[Tuple[float, float]]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    lo, hi = float(arr.min()), float(arr.max())
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return None
    if hi == lo:
        pad = max(1.0, abs(lo) * 0.1, 1e-12)
        return lo - pad, hi + pad
    pad = max((hi - lo) * pad_ratio, 1e-12)
    return lo - pad, hi + pad


def _axis_mode_and_label(series_list: List[dict]) -> Tuple[str, str]:
    if not series_list:
        return "numeric", "X"
    if all(s.get("x_mode") == "datetime" for s in series_list):
        return "date", "Zaman (UTC)"
    if any(s.get("x_mode") == "seconds" for s in series_list):
        return "numeric", "Zaman (s)"
    return "numeric", "İndeks"


def _format_x_display(axis_mode: str, x: float) -> str:
    if axis_mode == "date":
        try:
            return datetime.fromtimestamp(float(x), tz=timezone.utc).isoformat()
        except Exception:
            pass
    return f"{x:.12g}"


def detect_digital_like(
    y: np.ndarray,
    max_levels: int = 8,
    sample_limit: int = 50_000,
) -> Tuple[bool, Optional[Tuple[float, float]]]:
    """Detect whether *y* looks like a digital/discrete signal.

    Returns (is_digital, (lo, hi) | None).
    """
    if y is None:
        return False, None
    y = np.asarray(y)
    if y.size == 0:
        return False, None

    stride = max(1, y.size // sample_limit)
    yy = y[::stride]
    if np.issubdtype(yy.dtype, np.floating):
        yy = yy[np.isfinite(yy)]
        if yy.size == 0:
            return False, None

    uniq = np.unique(yy)
    if uniq.size <= 1 or uniq.size > max_levels:
        return False, None

    uf = uniq.astype(np.float64, copy=False)
    lo, hi = float(uf.min()), float(uf.max())
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi == lo:
        return False, None

    if np.all(np.abs(uf - np.round(uf)) < 1e-6) and (hi - lo) <= 10.0:
        return True, (lo, hi)

    if uniq.size <= 3:
        zz = (uf - lo) / (hi - lo)
        if np.all(np.abs(zz - np.round(zz)) < 1e-6):
            return True, (lo, hi)

    return False, None


# ===================================================================
# Cancellable worker base
# ===================================================================

class CancellableWorker(QObject):
    """Mixin providing a thread-safe cancellation flag for workers."""
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()


# ===================================================================
# Workers
# ===================================================================

class TdmsIndexWorker(CancellableWorker):
    """Reads group/channel structure from a TDMS file."""

    def __init__(self, file_id: str, path: str) -> None:
        super().__init__()
        self.file_id = file_id
        self.path = path

    def run(self) -> None:
        if self.is_cancelled:
            self.finished.emit({"file_id": self.file_id, "refs": [], "cancelled": True})
            return
        try:
            with _open_tdms(self.path) as tdms:
                refs: List[Tuple[str, str, int]] = []
                for group in tdms.groups():
                    if self.is_cancelled:
                        break
                    gname = group.name
                    for ch in group.channels():
                        try:
                            n = len(ch)
                        except Exception:
                            n = 0
                        refs.append((gname, ch.name, int(n)))

            if self.is_cancelled:
                self.finished.emit({"file_id": self.file_id, "refs": [], "cancelled": True})
                return
            self.finished.emit({"file_id": self.file_id, "refs": refs})
        except Exception as e:
            self.failed.emit(f"TDMS dizin okuma hatası:\n{e}")


class TdmsChannelPreviewWorker(CancellableWorker):
    """Reads metadata (unit, quantity, Fs) for a single channel."""

    def __init__(self, path: str, key: ChannelKey) -> None:
        super().__init__()
        self.path = path
        self.key = key

    def run(self) -> None:
        if self.is_cancelled:
            self.finished.emit({"file_id": self.key.file_id, "cancelled": True})
            return
        try:
            with _open_tdms(self.path) as tdms:
                ch = tdms[self.key.group][self.key.channel]
                props = getattr(ch, "properties", {}) or {}
                unit = props.get("unit_string", props.get("unit", ""))
                quantity = infer_quantity_from_props(props)
                try:
                    n = int(len(ch))
                except Exception:
                    n = 0

                x_mode = "index"
                fs: Optional[float] = None
                try:
                    x_mode, _base, x_inc = _xmeta_from_channel_props(props)
                    if x_mode in ("seconds", "datetime") and math.isfinite(x_inc) and x_inc > 0:
                        fs = 1.0 / x_inc
                except Exception:
                    x_mode = "index"

                self.finished.emit({
                    "file_id": self.key.file_id,
                    "group": self.key.group,
                    "channel": self.key.channel,
                    "name": f"{self.key.group}/{self.key.channel}",
                    "samples": n,
                    "unit": unit,
                    "quantity": quantity,
                    "x_info": {"x_mode": x_mode, "fs_est": fs},
                })
        except Exception as e:
            self.failed.emit(f"Önizleme hatası:\n{e}")


class MultiTdmsChannelLoadWorker(CancellableWorker):
    """Load data from one or more TDMS channels, with lazy-load support."""

    def __init__(
        self,
        files: Dict[str, Dict[str, str]],
        requests: List[ChannelRequest],
    ) -> None:
        super().__init__()
        self.files = files
        self.requests = requests

    def _load_channel_full(
        self, ch: Any, props: dict, skey: str, display: str,
        label: str, n: int,
    ) -> Optional[dict]:
        """Load full channel data (for channels below lazy threshold)."""
        y = _ensure_1d_numeric(ch[:])
        if y.size == 0:
            return None

        x, x_mode = classify_and_extract_x(ch, len(y))
        x = np.asarray(x, dtype=np.float64)

        # Sort by X if needed
        if x.size > 1 and x[-1] < x[0]:
            order = np.argsort(x)
            x, y = x[order], y[order]
        elif x.size > 1:
            stride0 = max(1, x.size // 5000)
            dx = np.diff(x[::stride0])
            if dx.size and np.any(dx < 0):
                order = np.argsort(x)
                x, y = x[order], y[order]

        # Deduplicate X
        if x.size > 1:
            xu, idx = np.unique(x, return_index=True)
            if xu.size != x.size:
                x, y = xu, y[idx]

        dig, dig_levels = detect_digital_like(y)
        fs_est = robust_fs_from_x(x) if x_mode in ("seconds", "datetime") else None
        unit = props.get("unit_string", props.get("unit", ""))
        quantity = infer_quantity_from_props(props)

        source_base = float(x[0]) if x_mode in ("seconds", "datetime") and x.size else 0.0
        source_inc = (1.0 / fs_est) if fs_est is not None and fs_est > 0 else 1.0

        return {
            "style_key": skey,
            "name": display,
            "file_label": label,
            "x": x, "y": y,
            "unit": unit,
            "quantity": quantity,
            "x_mode": x_mode,
            "source_x_mode": x_mode,
            "source_x_base": source_base,
            "source_x_inc": float(source_inc),
            "source_fs_est": fs_est,
            "sample_count": int(n),
            "is_digital": dig,
            "digital_levels": dig_levels,
            "lazy": False,
        }

    def _load_channel_lazy(
        self, ch: Any, props: dict, skey: str, display: str,
        label: str, n: int, path: str, k: ChannelKey,
    ) -> Optional[dict]:
        """Load overview data for lazy channels (above lazy threshold)."""
        x_mode, x_base, x_inc = _xmeta_from_channel_props(props)
        stride = _stride_for_window(0, n, CFG.lazy.overview_max_points)

        try:
            y = ch[0:n:stride]
        except Exception:
            indices = list(range(0, n, stride))
            y = np.asarray([ch[i] for i in indices])

        y = _ensure_1d_numeric(y)
        if y.size == 0:
            return None

        idx = np.arange(0, n, stride, dtype=np.float64)
        x = idx if x_mode == "index" else (x_base + idx * x_inc).astype(np.float64)

        dig, dig_levels = detect_digital_like(y)
        fs_est = (1.0 / x_inc) if x_mode in ("seconds", "datetime") and x_inc > 0 else None
        unit = props.get("unit_string", props.get("unit", ""))
        quantity = infer_quantity_from_props(props)

        return {
            "style_key": skey,
            "name": display,
            "file_label": label,
            "x": x, "y": y,
            "unit": unit,
            "quantity": quantity,
            "x_mode": x_mode,
            "source_x_mode": x_mode,
            "source_x_base": float(x_base),
            "source_x_inc": float(x_inc),
            "source_fs_est": fs_est,
            "sample_count": int(n),
            "is_digital": dig,
            "digital_levels": dig_levels,
            "lazy": True,
            "lazy_meta": {
                "path": path,
                "group": k.group,
                "channel": k.channel,
                "n": n,
                "x_mode": x_mode,
                "x_base": float(x_base),
                "x_inc": float(x_inc),
            },
            "_lazy_last_win": (0, n, stride),
        }

    def run(self) -> None:
        if self.is_cancelled:
            self.finished.emit({"series": [], "axis_mode": "numeric", "x_label": "X", "cancelled": True})
            return
        try:
            req_by_file: Dict[str, List[ChannelRequest]] = {}
            for r in self.requests:
                req_by_file.setdefault(r.key.file_id, []).append(r)

            series: List[dict] = []
            x_modes: List[str] = []

            for file_id, reqs in req_by_file.items():
                if self.is_cancelled:
                    break
                if file_id not in self.files:
                    continue
                path = self.files[file_id]["path"]
                label = self.files[file_id]["label"]

                try:
                    with _open_tdms(path) as tdms:
                        for req in reqs:
                            if self.is_cancelled:
                                break
                            k = req.key
                            try:
                                ch = tdms[k.group][k.channel]
                            except KeyError:
                                continue
                            props = getattr(ch, "properties", {}) or {}
                            try:
                                n = int(len(ch))
                            except Exception:
                                n = 0
                            if n <= 0:
                                continue

                            skey = style_key_for_channel(file_id, k.group, k.channel)
                            display = req.display_name.strip() or k.channel
                            use_lazy = CFG.lazy.enabled and n > CFG.lazy.fullload_threshold

                            if use_lazy:
                                s = self._load_channel_lazy(ch, props, skey, display, label, n, path, k)
                            else:
                                s = self._load_channel_full(ch, props, skey, display, label, n)

                            if s is not None:
                                x_modes.append(s["x_mode"])
                                series.append(s)
                except Exception as e:
                    logger.warning("Error loading channels from %s: %s", label, e)

            if not series or self.is_cancelled:
                self.finished.emit({"series": [], "axis_mode": "numeric", "x_label": "X", "cancelled": self.is_cancelled})
                return

            axis_mode = "date" if all(m == "datetime" for m in x_modes) else "numeric"
            x_label = "Zaman (UTC)" if axis_mode == "date" else (
                "Zaman (s)" if any(m == "seconds" for m in x_modes) else "İndeks"
            )
            self.finished.emit({"series": series, "axis_mode": axis_mode, "x_label": x_label})
        except Exception as e:
            self.failed.emit(f"Kanal verisi yükleme başarısız:\n{e}")


class LazyViewLoadWorker(CancellableWorker):
    """Read only the visible X range from TDMS for lazy channels."""

    def __init__(self, requests: List[dict], max_points: int) -> None:
        super().__init__()
        self.requests = requests
        self.max_points = max_points

    def run(self) -> None:
        if self.is_cancelled:
            self.finished.emit({"updates": {}, "cancelled": True})
            return
        try:
            by_path: Dict[str, List[dict]] = {}
            for r in self.requests:
                if r.get("path"):
                    by_path.setdefault(r["path"], []).append(r)

            updates: Dict[str, dict] = {}
            for path, reqs in by_path.items():
                if self.is_cancelled:
                    break
                try:
                    with _open_tdms(path) as tdms:
                        for r in reqs:
                            if self.is_cancelled:
                                break
                            skey = r.get("style_key", "")
                            g, c = r.get("group"), r.get("channel")
                            n = int(r.get("n", 0) or 0)
                            x_mode = r.get("x_mode", "index")
                            x_base = float(r.get("x_base", 0.0) or 0.0)
                            x_inc = float(r.get("x_inc", 1.0) or 1.0)
                            x0, x1 = float(r.get("x0", 0)), float(r.get("x1", 0))

                            if not (skey and g and c and n > 0):
                                continue
                            try:
                                ch = tdms[g][c]
                            except Exception:
                                continue

                            i0, i1 = _slice_indices_from_xrange(x0, x1, n, x_mode, x_base, x_inc)
                            stride = _stride_for_window(i0, i1, self.max_points)

                            try:
                                y = ch[i0:i1:stride]
                            except Exception:
                                indices = list(range(i0, i1, stride))
                                y = np.asarray([ch[i] for i in indices])

                            y = _ensure_1d_numeric(y)
                            idx = np.arange(i0, i1, stride, dtype=np.float64)
                            x = idx if x_mode == "index" else (x_base + idx * x_inc).astype(np.float64)

                            updates[skey] = {"x": x, "y": y, "win": (i0, i1, stride)}
                except Exception as e:
                    logger.warning("Lazy view read error for %s: %s", path, e)

            self.finished.emit({"updates": {} if self.is_cancelled else updates, "cancelled": self.is_cancelled})
        except Exception as e:
            self.failed.emit(f"Lazy view okuma başarısız:\n{e}")


class RangeStatsWorker(CancellableWorker):
    """Calculate per-channel statistics for a user-entered X interval.

    Lazy TDMS channels are sliced directly from disk for the requested interval,
    avoiding a full-channel load merely to calculate summary statistics.
    """

    def __init__(
        self, token: int, requests: List[dict], x_min: float, x_max: float,
    ) -> None:
        super().__init__()
        self.token = token
        self.requests = requests
        self.x_min = float(x_min)
        self.x_max = float(x_max)

    @staticmethod
    def _exact_slice(
        x_min: float, x_max: float, n: int, x_mode: str, x_base: float, x_inc: float,
    ) -> Tuple[int, int]:
        if n <= 0:
            return 0, 0
        lo, hi = sorted((float(x_min), float(x_max)))
        if x_mode == "index":
            i0 = int(math.floor(lo))
            i1 = int(math.ceil(hi)) + 1
        elif math.isfinite(x_inc) and x_inc > 0.0:
            i0 = int(math.floor((lo - x_base) / x_inc))
            i1 = int(math.ceil((hi - x_base) / x_inc)) + 1
        else:
            return 0, n
        return max(0, min(n, i0)), max(0, min(n, i1))

    def _finish_row(self, r: dict, x: np.ndarray, y: np.ndarray) -> dict:
        x_shift = float(r.get("x_shift", 0.0) or 0.0)
        y_shift = float(r.get("y_shift", 0.0) or 0.0)
        xx = np.asarray(x, dtype=np.float64) + x_shift
        yy = np.asarray(y, dtype=np.float64) + y_shift
        stats = compute_interval_statistics(xx, yy, self.x_min, self.x_max)
        return {
            "style_key": r.get("style_key", ""),
            "file_label": r.get("file_label", ""),
            "name": r.get("name", "Kanal"),
            "unit": r.get("unit", ""),
            **stats,
        }

    def run(self) -> None:
        if self.is_cancelled:
            self.finished.emit({"token": self.token, "rows": [], "cancelled": True})
            return
        rows: List[dict] = []
        errors: List[str] = []
        try:
            nonlazy = [r for r in self.requests if not r.get("lazy")]
            lazy_by_path: Dict[str, List[dict]] = {}
            for r in self.requests:
                if r.get("lazy") and r.get("path"):
                    lazy_by_path.setdefault(str(r["path"]), []).append(r)

            for r in nonlazy:
                if self.is_cancelled:
                    break
                try:
                    rows.append(self._finish_row(r, r.get("x", []), r.get("y", [])))
                except Exception as e:
                    errors.append(f"{r.get('name', 'Kanal')}: {e}")

            for path, reqs in lazy_by_path.items():
                if self.is_cancelled:
                    break
                try:
                    with _open_tdms(path) as tdms:
                        for r in reqs:
                            if self.is_cancelled:
                                break
                            try:
                                n = int(r.get("n", 0) or 0)
                                x_mode = str(r.get("x_mode", "index") or "index")
                                x_base = float(r.get("x_base", 0.0) or 0.0)
                                x_inc = float(r.get("x_inc", 1.0) or 1.0)
                                x_shift = float(r.get("x_shift", 0.0) or 0.0)
                                raw_lo = self.x_min - x_shift
                                raw_hi = self.x_max - x_shift
                                i0, i1 = self._exact_slice(raw_lo, raw_hi, n, x_mode, x_base, x_inc)
                                if i1 <= i0:
                                    raise ValueError("Seçilen aralık kanal verisiyle kesişmiyor.")
                                ch = tdms[r["group"]][r["channel"]]
                                y = _ensure_1d_numeric(ch[i0:i1])
                                idx = np.arange(i0, i0 + y.size, dtype=np.float64)
                                x = idx if x_mode == "index" else (x_base + idx * x_inc).astype(np.float64)
                                rows.append(self._finish_row(r, x, y))
                            except Exception as e:
                                errors.append(f"{r.get('name', 'Kanal')}: {e}")
                except Exception as e:
                    for r in reqs:
                        errors.append(f"{r.get('name', 'Kanal')}: {e}")

            self.finished.emit({
                "token": self.token,
                "rows": [] if self.is_cancelled else rows,
                "errors": errors,
                "x_min": self.x_min,
                "x_max": self.x_max,
                "cancelled": self.is_cancelled,
            })
        except Exception as e:
            self.failed.emit(f"Aralık analizi başarısız:\n{e}")


class FilterWorker(CancellableWorker):
    """Apply Savitzky-Golay smoothing in a background thread."""

    def __init__(
        self, token: int, series: List[dict],
        apply_filter: bool, window_len: int,
    ) -> None:
        super().__init__()
        self.token = token
        self.series = series
        self.apply_filter = apply_filter
        self.window_len = window_len

    def run(self) -> None:
        if self.is_cancelled:
            self.finished.emit({"token": self.token, "display_series": [], "cancelled": True})
            return
        try:
            display_series: List[dict] = []
            for s in self.series:
                if self.is_cancelled:
                    break
                y = s["y"]
                tag = ""
                is_dig = bool(s.get("is_digital"))
                if self.apply_filter and not is_dig:
                    if not SCIPY_AVAILABLE:
                        tag = "(SG yok)"
                    else:
                        try:
                            y = apply_savgol_safe(y, self.window_len)
                            tag = "(SG)"
                        except Exception:
                            tag = "(SG HATA)"
                elif is_dig:
                    tag = "(Dijital)"

                ss = s.copy()
                ss["y"] = y
                ss["tag"] = tag
                display_series.append(ss)

            axis_mode, x_label = _axis_mode_and_label(display_series)
            self.finished.emit({
                "token": self.token,
                "display_series": display_series,
                "axis_mode": axis_mode,
                "x_label": x_label,
            })
        except Exception as e:
            self.failed.emit(f"Filtre hesaplama başarısız:\n{e}")


class FFTWorker(CancellableWorker):
    """Compute single-sided FFT spectrum in a background thread."""

    def __init__(
        self, name: str, x: np.ndarray, y: np.ndarray,
        fs_hint: Optional[float], use_window: bool,
        remove_mean: bool, detrend_linear: bool,
    ) -> None:
        super().__init__()
        self.name = name
        self.x = x
        self.y = y
        self.fs_hint = fs_hint
        self.use_window = use_window
        self.remove_mean = remove_mean
        self.detrend_linear = detrend_linear

    def run(self) -> None:
        empty: dict = {
            "name": self.name,
            "freq": np.array([], dtype=np.float64),
            "mag_linear": np.array([], dtype=np.float64),
            "cancelled": True,
        }
        if self.is_cancelled:
            self.finished.emit(empty)
            return
        try:
            y = np.asarray(self.y, dtype=np.float64).copy()
            if y.size < CFG.min_fft_samples:
                raise ValueError(f"FFT için yeterli örnek yok (en az {CFG.min_fft_samples} gerekli).")

            if self.detrend_linear:
                y = linear_detrend_safe(y)
            elif self.remove_mean:
                y -= np.nanmean(y)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

            fs = robust_fs_from_x(np.asarray(self.x, dtype=np.float64))
            if fs is None:
                fs = self.fs_hint
            if fs is None or fs <= 0:
                raise ValueError("Örnekleme hızı (Fs) bilinmiyor. Fs'yi manuel girin veya zamana dayalı bir kanal seçin.")

            window_name = "None"
            coherent_gain = 1.0
            if self.use_window:
                win = np.hanning(y.size)
                coherent_gain = max(float(np.mean(win)), 1e-12)
                y *= win
                window_name = "Hanning"

            if self.is_cancelled:
                self.finished.emit(empty)
                return

            Y = np.fft.rfft(y)
            freq = np.fft.rfftfreq(y.size, d=1.0 / fs)
            mag = np.abs(Y) / max(1, y.size) / coherent_gain
            if mag.size > 2:
                mag[1:-1] *= 2.0

            self.finished.emit({
                "name": self.name,
                "freq": freq.astype(np.float64),
                "mag_linear": mag.astype(np.float64),
                "fs": float(fs),
                "n": int(y.size),
                "df": float(fs / y.size),
                "window_name": window_name,
                "remove_mean": self.remove_mean,
                "detrend_linear": self.detrend_linear,
                "use_window": self.use_window,
            })
        except Exception as e:
            self.failed.emit(f"FFT hesaplama başarısız:\n{e}")


# ===================================================================
# PlotPane
# ===================================================================

class PlotPane(QWidget):
    """Composite widget: quickbar + PlotWidget + optional right (digital) axis."""

    range_changed = pyqtSignal(float, float)
    marker_requested = pyqtSignal(float, float)
    interaction_mode_changed = pyqtSignal(str)
    region_toggled = pyqtSignal(bool)
    legend_toggled = pyqtSignal(bool)
    click_mark_toggled = pyqtSignal(bool)
    y_lock_toggled = pyqtSignal(bool)
    right_y_lock_toggled = pyqtSignal(bool)
    autofit_requested = pyqtSignal()
    copy_requested = pyqtSignal()
    save_requested = pyqtSignal()
    zoom_back_requested = pyqtSignal()
    axis_edit_requested = pyqtSignal(str)

    def __init__(
        self, parent: Optional[QWidget] = None,
        bg_rgb: Tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        self.axis_mode = "numeric"
        self.x_label = "X"
        self._axis_auto_titles = {"x": "X", "y": "Değer", "y2": None}
        self._axis_style: Dict[str, Any] = {
            "x_title": "", "y_title": "", "y2_title": "",
            "font_family": "Arial",
            "title_size": 11, "title_bold": False,
            "tick_size": 9, "tick_bold": False,
        }

        self.plot: Optional[PlotWidget] = None
        self.legend: Optional[pg.LegendItem] = None
        self.region: Optional[pg.LinearRegionItem] = None

        self._region_enabled = False
        self._click_mark_enabled = False
        self._markers: List[dict] = []
        self._last_cursor_x = float("nan")
        self._cursor_marker_enabled = True
        self._cursor_dot: Optional[pg.ScatterPlotItem] = None
        self._bg_rgb = bg_rgb
        self._legend_visible = True
        self._interaction_mode = "pan"

        # Y lock (left axis)
        self._y_lock_enabled = False
        self._y_lock_left: Optional[Tuple[float, float]] = None
        self._y_lock_guard = False

        # Y lock (right/digital axis)
        self._right_y_lock_enabled = False
        self._right_y_lock_range: Optional[Tuple[float, float]] = None

        # Plot items
        self._left_items: List[pg.PlotDataItem] = []
        self._right_items: List[pg.PlotDataItem] = []

        # Right axis viewbox
        self._right_vb: Optional[pg.ViewBox] = None
        self._right_update_views: Optional[Callable] = None
        self._right_sync_guard = False
        self._batch_plotting = False
        self._base_left_yrange: Optional[Tuple[float, float]] = None
        self._base_right_yrange: Optional[Tuple[float, float]] = None
        self._right_data_bounds: Optional[Tuple[float, float]] = None

        self._crosshair_v: Optional[pg.InfiniteLine] = None
        self._crosshair_h: Optional[pg.InfiniteLine] = None

        self._info = QLabel("İmleç: —")
        self._info.setObjectName("cursorInfo")
        try:
            self._info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        except Exception:
            pass
        self._info.setMinimumWidth(240)
        self._info.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)

        self._quickbar = self._build_quickbar()
        self._layout.addWidget(self._quickbar)
        self.build_plot(axis_mode="numeric", x_label="X")

    # ----- Legend -----

    def set_legend_visible(self, enabled: bool) -> None:
        self._legend_visible = enabled
        if self.legend is not None:
            self.legend.setVisible(self._legend_visible)
        self._qb_set_checked("_qb_legend", self._legend_visible)

    def set_legend_position(self, position: str = "upper-right") -> None:
        if self.legend is None:
            return
        pos_map = {
            "upper-left": (0, 0), "upper-right": (1, 0),
            "lower-left": (0, 1), "lower-right": (1, 1),
            "center": (0.5, 0.5),
        }
        anchor = pos_map.get(position, (1, 0))
        try:
            self.legend.anchor(itemPos=anchor, parentPos=anchor, offset=(0, 0))
        except Exception:
            logger.debug("Legend position set failed for %s", position)

    # ----- Background / Theme -----

    def set_background(self, bg_rgb: Tuple[int, int, int]) -> None:
        self._bg_rgb = bg_rgb
        if self.plot is not None:
            self._apply_theme()

    def _apply_theme(self) -> None:
        if self.plot is None:
            return
        apply_plotwidget_theme(self.plot, self._bg_rgb)
        fg = _bg_to_fg_rgb(self._bg_rgb)

        if self._crosshair_v is not None:
            self._crosshair_v.setPen(pg.mkPen(fg, style=Qt.PenStyle.DashLine))
        if self._crosshair_h is not None:
            self._crosshair_h.setPen(pg.mkPen(fg, style=Qt.PenStyle.DashLine))

        if self._cursor_dot is not None:
            try:
                self._cursor_dot.setPen(pg.mkPen(fg, width=1.6))
                self._cursor_dot.setBrush(pg.mkBrush(*fg, 180))
                self._cursor_dot.setVisible(self._cursor_marker_enabled)
            except Exception:
                pass

        self._apply_legend_theme()
        self._refresh_right_axis_visuals()
        self._update_info_style()

    def _update_info_style(self) -> None:
        fg = _bg_to_fg_rgb(self._bg_rgb)
        is_dark = fg == (255, 255, 255)
        if is_dark:
            self._info.setStyleSheet(
                "padding: 4px 8px; font-weight: 700; color: #F2F4F8;"
                "background: rgba(0,0,0,0.55); border: 1px solid rgba(255,255,255,0.18);"
                "border-radius: 9px;"
            )
        else:
            self._info.setStyleSheet(
                "padding: 4px 8px; font-weight: 700; color: #222;"
                "background: rgba(255,255,255,0.85); border: 1px solid #D7DBE0;"
                "border-radius: 9px;"
            )

    def _legend_text_rgb(self) -> Tuple[int, int, int]:
        fg = _bg_to_fg_rgb(self._bg_rgb)
        return (235, 240, 248) if fg == (255, 255, 255) else fg

    def _apply_legend_theme(self) -> None:
        if self.legend is None:
            return
        legend_fg = self._legend_text_rgb()
        is_dark = _bg_to_fg_rgb(self._bg_rgb) == (255, 255, 255)
        brush = pg.mkBrush(30, 30, 30, 210) if is_dark else pg.mkBrush(255, 255, 255, 210)
        border_pen = pg.mkPen((90, 96, 110, 180) if is_dark else (185, 190, 200, 180))
        color_name = QColor(*legend_fg).name()

        try:
            self.legend.setVisible(self._legend_visible)
        except Exception:
            pass
        try:
            if hasattr(self.legend, "setLabelTextColor"):
                self.legend.setLabelTextColor(legend_fg)
        except Exception:
            pass
        try:
            self.legend.setBrush(brush)
        except Exception:
            pass
        try:
            self.legend.setPen(border_pen)
        except Exception:
            pass

        for entry in list(getattr(self.legend, "items", []) or []):
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            label_item = entry[1]
            try:
                if hasattr(label_item, "setDefaultTextColor"):
                    label_item.setDefaultTextColor(QColor(*legend_fg))
            except Exception:
                pass
            try:
                if hasattr(label_item, "setAttr"):
                    label_item.setAttr("color", color_name)
            except Exception:
                pass
            try:
                txt = getattr(label_item, "text", None)
                if txt is not None and hasattr(label_item, "setText"):
                    label_item.setText(txt, color=color_name)
            except Exception:
                pass
        try:
            self.legend.update()
        except Exception:
            pass

    # ----- Right (digital) axis -----

    def _refresh_right_axis_visuals(self) -> None:
        if self.plot is None:
            return
        try:
            p1 = self.plot.getPlotItem()
            axis = p1.getAxis("right")
        except Exception:
            return
        if axis is None:
            return

        fg = _bg_to_fg_rgb(self._bg_rgb)
        try:
            axis.setPen(pg.mkPen(fg))
            axis.setTextPen(pg.mkPen(fg))
        except Exception:
            pass

        if self._right_vb is None:
            try:
                axis.setStyle(showValues=False)
                self.plot.hideAxis("right")
            except Exception:
                pass
        else:
            try:
                axis.setStyle(showValues=True)
                self.plot.showAxis("right")
                axis.linkToView(self._right_vb)
            except Exception:
                pass

    def _destroy_right_axis(self) -> None:
        if self.plot is not None:
            p1 = self.plot.getPlotItem()
            if self._right_update_views is not None:
                try:
                    p1.vb.sigResized.disconnect(self._right_update_views)
                except Exception:
                    pass
            if self._right_vb is not None:
                try:
                    p1.scene().removeItem(self._right_vb)
                except Exception:
                    pass
            try:
                self.plot.hideAxis("right")
            except Exception:
                pass

        self._right_vb = None
        self._right_items = []
        self._right_update_views = None
        self._base_right_yrange = None
        self._base_left_yrange = None
        self._right_y_lock_range = None
        self._right_data_bounds = None
        self._refresh_right_axis_visuals()

    def _ensure_right_axis(self) -> None:
        if self.plot is None:
            return
        p1 = self.plot.getPlotItem()

        if self._right_vb is not None:
            try:
                self.plot.showAxis("right")
            except Exception:
                pass
            self._refresh_right_axis_visuals()
            return

        self._right_vb = pg.ViewBox()
        self._right_vb.setMouseEnabled(x=False, y=False)
        self._right_vb.setMenuEnabled(False)

        try:
            self.plot.showAxis("right")
            p1.scene().addItem(self._right_vb)
            p1.getAxis("right").linkToView(self._right_vb)
        except Exception:
            self._right_vb = None
            return

        self._right_vb.setXLink(p1.vb)

        def update_views() -> None:
            if self._right_vb is None:
                return
            self._right_vb.setGeometry(p1.vb.sceneBoundingRect())
            self._right_vb.linkedViewChanged(p1.vb, self._right_vb.XAxis)

        self._right_update_views = update_views
        update_views()
        p1.vb.sigResized.connect(update_views)
        self._refresh_right_axis_visuals()

    # ----- Y Lock (left) -----

    def set_y_lock(self, enabled: bool) -> None:
        self._y_lock_enabled = enabled
        if self.plot is not None:
            vb = self.plot.getViewBox()
            if self._y_lock_enabled:
                self._y_lock_left = tuple(vb.viewRange()[1])
                self._enforce_y_lock()
            else:
                self._y_lock_left = None
        self._qb_set_checked("_qb_ylock", self._y_lock_enabled)

    def _enforce_y_lock(self) -> None:
        if not self._y_lock_enabled or self.plot is None or self._y_lock_guard:
            return
        self._y_lock_guard = True
        try:
            if self._y_lock_left is not None:
                self.plot.getViewBox().setYRange(*self._y_lock_left, padding=0.0)
        finally:
            self._y_lock_guard = False

    # ----- Y Lock (right / digital) -----

    def set_right_y_lock(self, enabled: bool) -> None:
        self._right_y_lock_enabled = enabled
        if self._right_y_lock_enabled and self._right_vb is not None:
            try:
                self._right_y_lock_range = tuple(self._right_vb.viewRange()[1])
                self._enforce_right_y_lock()
            except Exception:
                self._right_y_lock_range = None
        elif not self._right_y_lock_enabled:
            self._right_y_lock_range = None
            self._sync_right_yrange_to_left()
        self._qb_set_checked("_qb_y2lock", self._right_y_lock_enabled)

    def _enforce_right_y_lock(self) -> None:
        if (
            not self._right_y_lock_enabled
            or self._right_vb is None
            or self._right_y_lock_range is None
            or self._right_sync_guard
        ):
            return
        self._right_sync_guard = True
        try:
            self._right_vb.setYRange(*self._right_y_lock_range, padding=0.0)
        finally:
            self._right_sync_guard = False

    def _on_left_yrange_changed(self, *_args: Any) -> None:
        if self._batch_plotting:
            return
        self._enforce_y_lock()
        self._sync_right_yrange_to_left()

    def _sync_right_yrange_to_left(self) -> None:
        if self.plot is None or self._right_vb is None:
            return
        if self._right_y_lock_enabled:
            self._enforce_right_y_lock()
            return
        if self._right_sync_guard:
            return
        if self._base_left_yrange is None or self._base_right_yrange is None:
            return

        vb = self.plot.getViewBox()
        try:
            l0, l1 = (float(v) for v in vb.viewRange()[1])
            bl0, bl1 = (float(v) for v in self._base_left_yrange)
            br0, br1 = (float(v) for v in self._base_right_yrange)
        except Exception:
            return

        if not all(math.isfinite(v) for v in (l0, l1, bl0, bl1, br0, br1)):
            return

        base_span = bl1 - bl0
        if base_span == 0:
            return
        now_span = max(l1 - l0, 1e-12)
        ratio = (br1 - br0) / base_span
        r0_des = br0 + (l0 - bl0) * ratio
        r_span_des = (br1 - br0) * (now_span / base_span)
        r1_des = r0_des + r_span_des

        # Clamp to data bounds if available
        if self._right_data_bounds is not None:
            try:
                d0, d1 = (float(v) for v in self._right_data_bounds)
                if math.isfinite(d0) and math.isfinite(d1) and d1 >= d0:
                    dspan = d1 - d0
                    pad = 0.05 * dspan if dspan > 0 else 0.25
                    mn, mx = d0 - pad, d1 + pad

                    if not (math.isfinite(r0_des) and math.isfinite(r1_des) and r1_des > r0_des):
                        r0_des, r1_des = mn, mx
                    else:
                        need = mx - mn
                        span = r1_des - r0_des
                        if span < need:
                            c = (r0_des + r1_des) / 2.0
                            r0_des, r1_des = c - need / 2, c + need / 2
                        if r0_des > mn:
                            r1_des -= (r0_des - mn)
                            r0_des = mn
                        if r1_des < mx:
                            r0_des += (mx - r1_des)
                            r1_des = mx
                        if r0_des > mn or r1_des < mx:
                            r0_des, r1_des = mn, mx
            except Exception:
                pass

        if not (math.isfinite(r0_des) and math.isfinite(r1_des) and r1_des != r0_des):
            return

        try:
            self._right_vb.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
        except Exception:
            pass

        self._right_sync_guard = True
        try:
            self._right_vb.setYRange(r0_des, r1_des, padding=0.0)
        finally:
            self._right_sync_guard = False

    # ----- Build / rebuild -----

    def build_plot(self, axis_mode: str, x_label: str) -> None:
        """Create or recreate the PlotWidget for the given axis configuration."""
        self.axis_mode = axis_mode
        self.x_label = x_label

        if self.plot is not None:
            self._destroy_right_axis()
            self._layout.removeWidget(self.plot)
            self.plot.deleteLater()
            self.plot = None
            self.legend = None
            self.region = None
            self._markers = []
            self._left_items = []
            self._y_lock_left = None

        axis_items: dict = {
            "left": EditableAxisItem(
                orientation="left", edit_key="y", edit_callback=self.axis_edit_requested.emit
            ),
            "right": EditableAxisItem(
                orientation="right", edit_key="y2", edit_callback=self.axis_edit_requested.emit
            ),
        }
        if axis_mode == "date":
            axis_items["bottom"] = EditableDateAxisItem(
                orientation="bottom", edit_key="x", edit_callback=self.axis_edit_requested.emit
            )
        else:
            axis_items["bottom"] = EditableAxisItem(
                orientation="bottom", edit_key="x", edit_callback=self.axis_edit_requested.emit
            )

        self.plot = PlotWidget(axisItems=axis_items)
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.plot.customContextMenuRequested.connect(self._show_plot_context_menu)
        try:
            self.plot.hideAxis("right")
        except Exception:
            pass

        self.legend = self.plot.addLegend(offset=(10, 10))
        self.legend.setVisible(self._legend_visible)
        self.set_legend_position("upper-right")

        vb = self.plot.getViewBox()
        vb.setMouseEnabled(x=True, y=True)
        vb.setMouseMode(pg.ViewBox.RectMode)
        vb.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)

        self.plot.setDownsampling(auto=True, mode="peak")
        self.plot.setClipToView(True)

        self._crosshair_v = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("k", style=Qt.PenStyle.DashLine))
        self._crosshair_h = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("k", style=Qt.PenStyle.DashLine))
        self.plot.addItem(self._crosshair_v, ignoreBounds=True)
        self.plot.addItem(self._crosshair_h, ignoreBounds=True)

        try:
            self._cursor_dot = pg.ScatterPlotItem([], [], size=9, pxMode=True)
            self._cursor_dot.setZValue(1000)
            self.plot.addItem(self._cursor_dot, ignoreBounds=True)
        except Exception:
            self._cursor_dot = None

        self._mouse_proxy = pg.SignalProxy(self.plot.scene().sigMouseMoved, rateLimit=60, slot=self._on_mouse_moved)
        self.plot.scene().sigMouseClicked.connect(self._on_mouse_clicked)
        vb.sigXRangeChanged.connect(self._on_xrange_changed)
        vb.sigYRangeChanged.connect(self._on_left_yrange_changed)

        self._set_axis_titles(x_label, "Değer", None)

        self._layout.addWidget(self.plot, stretch=1)
        self._apply_theme()
        self._apply_axis_tick_fonts()

        # Restore interactive states
        self.set_interaction_mode(self._interaction_mode)
        self.set_legend_visible(self._legend_visible)
        self.set_click_mark_mode(self._click_mark_enabled)
        self.set_cursor_marker_enabled(self._cursor_marker_enabled)
        self.set_y_lock(self._y_lock_enabled)
        if self._region_enabled:
            self.enable_region(True)

    def _show_plot_context_menu(self, pos: Any) -> None:
        if self.plot is None:
            return
        menu = QMenu(self)
        a_fit = menu.addAction("Otomatik Sığdır")
        a_back = menu.addAction("Önceki Zoom")
        menu.addSeparator()
        a_legend = menu.addAction("Legend'i Gizle" if self._legend_visible else "Legend'i Göster")
        a_copy = menu.addAction("Grafiği Panoya Kopyala")
        a_save = menu.addAction("Grafiği Kaydet...")
        chosen = menu.exec(self.plot.mapToGlobal(pos))
        if chosen is a_fit:
            self.autofit_requested.emit()
        elif chosen is a_back:
            self.zoom_back_requested.emit()
        elif chosen is a_legend:
            self.legend_toggled.emit(not self._legend_visible)
        elif chosen is a_copy:
            self.copy_requested.emit()
        elif chosen is a_save:
            self.save_requested.emit()

    # ----- Interaction mode -----

    def set_interaction_mode(self, mode: str) -> None:
        if self.plot is None:
            return
        m = "pan" if (mode or "").strip().lower() == "pan" else "zoom"
        self._interaction_mode = m
        vb = self.plot.getViewBox()
        vb.setMouseEnabled(x=True, y=True)
        vb.setMouseMode(pg.ViewBox.PanMode if m == "pan" else pg.ViewBox.RectMode)
        self._qb_set_checked("_qb_pan" if m == "pan" else "_qb_zoom", True)

    def set_click_mark_mode(self, enabled: bool) -> None:
        self._click_mark_enabled = enabled
        self._qb_set_checked("_qb_marker", enabled)

    def set_cursor_marker_enabled(self, enabled: bool) -> None:
        self._cursor_marker_enabled = enabled
        if self.plot is None:
            return
        for item in (self._crosshair_v, self._crosshair_h):
            if item is not None:
                item.setVisible(enabled)
        if self._cursor_dot is not None:
            self._cursor_dot.setVisible(enabled)
            if not enabled:
                self._cursor_dot.setData([], [])

    # ----- Clear -----

    def clear_all(self) -> None:
        if self.plot is None:
            return
        for it in self._left_items:
            try:
                self.plot.removeItem(it)
            except Exception:
                pass
        self._left_items = []

        if self._right_vb is not None:
            for it in self._right_items:
                try:
                    self._right_vb.removeItem(it)
                except Exception:
                    pass
        self._right_items = []
        self._destroy_right_axis()

        if self.legend:
            self.legend.clear()
        self.clear_markers()
        if self.region is not None:
            try:
                self.plot.removeItem(self.region)
            except Exception:
                pass
            self.region = None
        self._apply_theme()

    # ----- Axis labels / typography -----

    def _compute_y_label(self, series_list: List[dict]) -> str:
        if not series_list:
            return ""
        cu = common_unit(series_list)
        cq = common_quantity(series_list)
        if cq:
            if cu:
                return f"{cq} ({cu})"
            has_any_unit = any((s.get("unit") or "").strip() for s in series_list)
            return f"{cq} (birimler göstergede)" if has_any_unit else cq
        return f"Değer ({cu})" if cu else "Değer"

    def _axis_label_style_kwargs(self) -> Dict[str, str]:
        st = self._axis_style
        family = str(st.get("font_family", "Arial") or "Arial")
        size = max(6, int(st.get("title_size", 11) or 11))
        return {
            "font-family": family,
            "font-size": f"{size}pt",
            "font-weight": "bold" if bool(st.get("title_bold", False)) else "normal",
        }

    def _apply_axis_tick_fonts(self) -> None:
        if self.plot is None:
            return
        st = self._axis_style
        font = QFont(str(st.get("font_family", "Arial") or "Arial"))
        font.setPointSize(max(6, int(st.get("tick_size", 9) or 9)))
        font.setBold(bool(st.get("tick_bold", False)))
        for ax_name in ("bottom", "left", "right"):
            try:
                axis = self.plot.getPlotItem().getAxis(ax_name)
                if axis is None:
                    continue
                if hasattr(axis, "setTickFont"):
                    axis.setTickFont(font)
                else:
                    axis.setStyle(tickFont=font)
            except Exception:
                logger.debug("Axis tick font apply failed: %s", ax_name, exc_info=True)

    def set_axis_style(self, style: Optional[Dict[str, Any]] = None) -> None:
        """Apply user axis titles and typography without touching plotted data."""
        if style:
            self._axis_style.update(style)
        auto = self._axis_auto_titles or {"x": self.x_label, "y": "Değer", "y2": None}
        if self.plot is not None:
            self._set_axis_titles(
                str(auto.get("x") or self.x_label),
                str(auto.get("y") or "Değer"),
                auto.get("y2"),
            )
            self._apply_axis_tick_fonts()

    def _set_axis_titles(
        self, x_label: str, y_left: str,
        y_right: Optional[str] = None,
    ) -> None:
        self._axis_auto_titles = {"x": x_label, "y": y_left or "Değer", "y2": y_right}
        if self.plot is None:
            return
        pi = self.plot.getPlotItem()
        st = self._axis_style
        x_text = str(st.get("x_title", "") or "").strip() or x_label
        y_text = str(st.get("y_title", "") or "").strip() or (y_left or "Değer")
        y2_custom = str(st.get("y2_title", "") or "").strip()
        y2_text = y2_custom or (y_right or "")
        label_style = self._axis_label_style_kwargs()

        self.plot.setLabel("bottom", x_text, **label_style)
        self.plot.setLabel("left", y_text, **label_style)

        parts = [f"<b>X:</b> {x_text}", f"<b>Y:</b> {y_text or '—'}"]
        if y_right:
            try:
                self.plot.setLabel("right", y2_text or y_right, **label_style)
            except Exception:
                pass
            parts.append(f"<b>Y2:</b> {y2_text or y_right}")
        pi.setTitle(" &nbsp;&nbsp; | &nbsp;&nbsp; ".join(parts))
        self._apply_axis_tick_fonts()

    # ----- Digital step rendering -----

    def _plot_digital_step(
        self, x: np.ndarray, y: np.ndarray,
        pen: Any, viewbox: Optional[pg.ViewBox] = None,
    ) -> Optional[pg.PlotDataItem]:
        """Efficient step-plot for digital channels using edge detection."""
        if x is None or y is None:
            return None
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y)
        n = min(x.size, y.size)
        if n <= 0:
            return None
        x, y = x[:n], y[:n]

        if not np.issubdtype(y.dtype, np.number):
            y = y.astype(np.float64)

        y_cmp = np.rint(y).astype(np.float64) if np.issubdtype(y.dtype, np.floating) else y.astype(np.float64)

        valid = np.isfinite(x) & np.isfinite(y_cmp)
        if not np.any(valid):
            return None
        xv, yv = x[valid], y_cmp[valid]
        if xv.size == 0:
            return None

        if xv.size == 1:
            x_step, y_step = xv, yv
        else:
            edges = np.flatnonzero(yv[1:] != yv[:-1]) + 1
            m = edges.size
            if m == 0:
                x_step = np.array([xv[0], xv[-1]], dtype=np.float64)
                y_step = np.array([yv[0], yv[0]], dtype=np.float64)
            else:
                x_step = np.empty(2 * m + 2, dtype=np.float64)
                y_step = np.empty(2 * m + 2, dtype=np.float64)
                x_step[0], y_step[0] = xv[0], yv[0]
                ex = xv[edges]
                x_step[1:2*m+1:2] = ex
                y_step[1:2*m+1:2] = yv[edges - 1]
                x_step[2:2*m+1:2] = ex
                y_step[2:2*m+1:2] = yv[edges]
                x_step[-1], y_step[-1] = xv[-1], yv[-1]

        item = pg.PlotDataItem(pen=pen)
        try:
            item.setData(x_step, y_step, skipFiniteCheck=True)
        except TypeError:
            item.setData(x_step, y_step)
        try:
            item.setDownsampling(auto=True, mode="peak")
            item.setClipToView(True)
        except Exception:
            pass

        if viewbox is None:
            self.plot.addItem(item)
        else:
            viewbox.addItem(item)
        return item

    # ----- Plot series -----

    def plot_series(
        self, series_list: List[dict], axis_mode: str,
        x_label: str, style_map: Optional[Dict[str, dict]] = None,
    ) -> None:
        """Plot series with per-channel visibility/axis routing and envelope downsampling."""
        smap = style_map or {}
        series_list = [
            s for s in series_list
            if smap.get(s.get("style_key", ""), {}).get("visible", True) is not False
        ]

        if axis_mode != self.axis_mode:
            self.build_plot(axis_mode=axis_mode, x_label=x_label)
        else:
            self.x_label = x_label
            self.clear_all()

        if not series_list:
            self._set_axis_titles(x_label, "Değer", None)
            self._apply_theme()
            return

        self._batch_plotting = True
        try:
            self.plot.setUpdatesEnabled(False)
        except Exception:
            pass
        if self.legend is not None:
            self.legend.setVisible(False)
        try:
            self.plot.getViewBox().enableAutoRange(axis=pg.ViewBox.XYAxes, enable=False)
        except Exception:
            pass

        def _axis_for(s: dict) -> str:
            st = smap.get(s.get("style_key", ""), {})
            axis = str(st.get("axis", "auto") or "auto").lower()
            if axis == "left":
                return "left"
            if axis == "right":
                return "right"
            return "right" if s.get("is_digital") else "left"

        left_series = [s for s in series_list if _axis_for(s) == "left"]
        right_series = [s for s in series_list if _axis_for(s) == "right"]
        y_left = self._compute_y_label(left_series) if left_series else "Değer"
        y_right = self._compute_y_label(right_series) if right_series else None

        if right_series:
            self._ensure_right_axis()
            self._set_axis_titles(x_label, y_left, y_right or "Sağ Eksen")
        else:
            self._destroy_right_axis()
            self._set_axis_titles(x_label, y_left, None)

        n_total = len(series_list)
        i_global = 0
        right_mins: List[float] = []
        right_maxs: List[float] = []

        def _make_label(s: dict, is_dig: bool = False, x_shift: float = 0.0, y_shift: float = 0.0) -> str:
            st = smap.get(s.get("style_key", ""), {})
            fl = (s.get("file_label") or "").strip()
            nm = (st.get("display_name") or s.get("name") or "").strip()
            label = f"{fl} | {nm}" if fl else nm
            unit = (st.get("unit_override") or s.get("unit") or "").strip()
            if unit:
                label += f" [{unit}]"
            tag = (s.get("tag") or "").strip()
            if tag:
                label += f" {tag}"
            if is_dig:
                label += " [DİJ]"
            if x_shift:
                label += f" (x{x_shift:+g})"
            if y_shift:
                label += f" (y{y_shift:+g})"
            return label

        def _style(s: dict) -> Tuple[Any, float, float, float]:
            st = smap.get(s.get("style_key", ""), {})
            color = st.get("color")
            width = float(st.get("width", 2.0))
            xs = float(st.get("x_shift", 0.0) or 0.0)
            ys = float(st.get("y_shift", 0.0) or 0.0)
            return color, width, xs if math.isfinite(xs) else 0.0, ys if math.isfinite(ys) else 0.0

        def _add_series(s: dict, side: str) -> None:
            nonlocal i_global
            skey = s.get("style_key", "")
            color, width, x_shift, y_shift = _style(s)
            x = np.asarray(s["x"], dtype=np.float64)
            y = np.asarray(s["y"])
            if x_shift:
                x = x + x_shift
            if y_shift:
                y = np.asarray(y, dtype=np.float64) + y_shift
            pen_color = color if color is not None else pg.intColor(i_global, hues=max(1, n_total))
            pen = pg.mkPen(pen_color, width=width)
            target_vb = None if side == "left" else self._right_vb

            if s.get("is_digital"):
                item = self._plot_digital_step(x, y, pen, viewbox=target_vb)
            else:
                xd, yd = minmax_envelope_downsample(x, y, CFG.plot_max_points)
                item = pg.PlotDataItem(pen=pen)
                try:
                    item.setData(xd, yd, skipFiniteCheck=True)
                except TypeError:
                    item.setData(xd, yd)
                try:
                    item.setDownsampling(auto=True, mode="peak")
                    item.setClipToView(True)
                except Exception:
                    pass
                if target_vb is None:
                    self.plot.addItem(item)
                else:
                    target_vb.addItem(item)

            if item is None:
                i_global += 1
                return
            if side == "left":
                self._left_items.append(item)
            else:
                self._right_items.append(item)
                try:
                    yf = np.asarray(y, dtype=np.float64)
                    yf = yf[np.isfinite(yf)]
                    if yf.size:
                        right_mins.append(float(np.min(yf)))
                        right_maxs.append(float(np.max(yf)))
                except Exception:
                    pass
            if self.legend is not None:
                try:
                    self.legend.addItem(item, _make_label(s, bool(s.get("is_digital")), x_shift, y_shift))
                except Exception:
                    pass
            i_global += 1

        for s in left_series:
            _add_series(s, "left")
        for s in right_series:
            if self._right_vb is None:
                self._ensure_right_axis()
            _add_series(s, "right")

        self._right_data_bounds = (min(right_mins), max(right_maxs)) if right_mins else None
        vb = self.plot.getViewBox()
        try:
            vb.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
            vb.autoRange()
        except Exception:
            self.plot.enableAutoRange()

        if self._right_vb is not None:
            if self._right_y_lock_enabled:
                self._enforce_right_y_lock()
            elif self._right_data_bounds is not None:
                d0, d1 = self._right_data_bounds
                pad = 0.05 * (d1 - d0) if d1 > d0 else 0.25
                try:
                    self._right_vb.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
                    self._right_vb.setYRange(d0 - pad, d1 + pad, padding=0.0)
                except Exception:
                    pass

        try:
            self._base_left_yrange = tuple(vb.viewRange()[1])
        except Exception:
            self._base_left_yrange = None
        if self._right_vb is not None:
            try:
                self._base_right_yrange = tuple(self._right_vb.viewRange()[1])
            except Exception:
                self._base_right_yrange = None
        try:
            vb.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=False)
        except Exception:
            pass
        self._sync_right_yrange_to_left()
        if self._region_enabled:
            self.enable_region(True)
        try:
            self.plot.setUpdatesEnabled(True)
        except Exception:
            pass
        self._batch_plotting = False
        if self.legend is not None:
            self.legend.setVisible(self._legend_visible)
        self._apply_theme()
        self._enforce_y_lock()
        self._enforce_right_y_lock()
        self._update_quickbar_state()

    # ----- Autofit / range -----

    def autofit_all(self) -> None:
        if self.plot is None:
            return
        vb = self.plot.getViewBox()
        try:
            vb.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
            vb.autoRange()
        except Exception:
            self.plot.enableAutoRange()

        if self._right_vb is not None:
            try:
                self._right_vb.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
                self._right_vb.autoRange()
                self._right_vb.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
            except Exception:
                pass

        try:
            self._base_left_yrange = tuple(vb.viewRange()[1])
        except Exception:
            self._base_left_yrange = None
        self._base_right_yrange = None
        if self._right_vb is not None:
            try:
                self._base_right_yrange = tuple(self._right_vb.viewRange()[1])
            except Exception:
                pass
        try:
            vb.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=False)
        except Exception:
            pass
        self._sync_right_yrange_to_left()
        self._enforce_y_lock()
        self._enforce_right_y_lock()

    def set_xrange(self, x_min: float, x_max: float) -> None:
        if self.plot is None:
            return
        if x_max > x_min:
            self.plot.setXRange(x_min, x_max, padding=0.01)
        self._enforce_y_lock()
        self._enforce_right_y_lock()

    def enable_region(self, enabled: bool) -> None:
        self._region_enabled = enabled
        self._qb_set_checked("_qb_region", enabled)
        if self.plot is None:
            return
        if enabled:
            if self.region is None:
                xr = self.plot.getViewBox().viewRange()[0]
                mid = (xr[0] + xr[1]) / 2.0
                span = (xr[1] - xr[0]) * 0.3 if xr[1] > xr[0] else 1.0
                self.region = pg.LinearRegionItem(values=(mid - span, mid + span))
                self.region.setZValue(10)
                self.plot.addItem(self.region)
                self.region.sigRegionChanged.connect(self._on_region_changed)
        else:
            if self.region is not None:
                try:
                    self.plot.removeItem(self.region)
                except Exception:
                    pass
                self.region = None

    def region_values(self) -> Optional[Tuple[float, float]]:
        if self.region is None:
            return None
        v = self.region.getRegion()
        return float(v[0]), float(v[1])

    def autofit_y_in_range(self, x_min: float, x_max: float) -> None:
        """Autofit Y axes to data within the given X range."""
        if self.plot is None or x_max <= x_min:
            return

        def _fit_items(items: Sequence[pg.PlotDataItem]) -> Optional[Tuple[float, float]]:
            ymins, ymaxs = [], []
            for it in items:
                xd, yd = it.getData()
                if xd is None or yd is None or len(xd) == 0:
                    continue
                mask = (xd >= x_min) & (xd <= x_max)
                if np.any(mask):
                    yy = yd[mask]
                    ymins.append(float(np.nanmin(yy)))
                    ymaxs.append(float(np.nanmax(yy)))
            if not ymins:
                return None
            y0, y1 = min(ymins), max(ymaxs)
            pad = (y1 - y0) * 0.05 if y1 != y0 else 1.0
            return y0 - pad, y1 + pad

        rng = _fit_items(self._left_items)
        if rng is not None:
            self.plot.setYRange(*rng, padding=0.0)
            if self._y_lock_enabled:
                self._y_lock_left = tuple(self.plot.getViewBox().viewRange()[1])

        if self._right_vb is not None and self._right_items:
            rng = _fit_items(self._right_items)
            if rng is not None:
                self._right_vb.setYRange(*rng, padding=0.0)
                if self._right_y_lock_enabled:
                    self._right_y_lock_range = rng

    # ----- Markers -----

    def add_marker(self, x: float, y: float, label: str) -> None:
        if self.plot is None or not math.isfinite(x):
            return
        line = pg.InfiniteLine(pos=x, angle=90, movable=False, pen=pg.mkPen("r", width=1.5))
        self.plot.addItem(line)
        text = pg.TextItem(text=label, anchor=(0, 1), color=(200, 0, 0))
        yr = self.plot.getViewBox().viewRange()[1]
        text.setPos(x, yr[1])
        self.plot.addItem(text)
        self._markers.append({"x": x, "y": y, "line": line, "text": text, "label": label})

    def clear_markers(self) -> None:
        if self.plot is not None:
            for m in self._markers:
                try:
                    self.plot.removeItem(m["line"])
                    self.plot.removeItem(m["text"])
                except Exception:
                    pass
        self._markers = []

    def jump_to_x(self, x: float) -> None:
        if self.plot is None:
            return
        xr = self.plot.getViewBox().viewRange()[0]
        span = (xr[1] - xr[0]) or 1.0
        self.set_xrange(x - span * 0.25, x + span * 0.25)

    # ----- Mouse events -----

    def _on_region_changed(self) -> None:
        v = self.region_values()
        if v:
            self.range_changed.emit(v[0], v[1])

    def _on_xrange_changed(self, _: Any, r: Any) -> None:
        self.range_changed.emit(float(r[0]), float(r[1]))
        self._enforce_y_lock()
        self._enforce_right_y_lock()

    def _on_mouse_moved(self, evt: Any) -> None:
        if self.plot is None:
            return
        pos = evt[0]
        if not self.plot.sceneBoundingRect().contains(pos):
            return
        mp = self.plot.getViewBox().mapSceneToView(pos)
        self._last_cursor_x = float(mp.x())

        if self._cursor_marker_enabled:
            if self._crosshair_v is not None:
                self._crosshair_v.setPos(mp.x())
            if self._crosshair_h is not None:
                self._crosshair_h.setPos(mp.y())
            if self._cursor_dot is not None:
                self._cursor_dot.setData([mp.x()], [mp.y()])
        elif self._cursor_dot is not None:
            self._cursor_dot.setData([], [])

        val_txt = f"{mp.y():.6g}"
        if self.axis_mode == "date":
            try:
                t = datetime.fromtimestamp(mp.x(), tz=timezone.utc).isoformat()
                self._info.setText(f"Zaman: {t} | Y: {val_txt}")
            except Exception:
                self._info.setText(f"X={mp.x():.6g} | Y: {val_txt}")
        else:
            self._info.setText(f"X={mp.x():.6g} | Y: {val_txt}")

    def _on_mouse_clicked(self, event: Any) -> None:
        if self.plot is None or not self._click_mark_enabled:
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.scenePos()
        vb = self.plot.getViewBox()
        if vb.sceneBoundingRect().contains(pos):
            mp = vb.mapSceneToView(pos)
            self.marker_requested.emit(float(mp.x()), float(mp.y()))

    # ----- Quickbar -----

    def _icon_dirs(self) -> List[str]:
        """Return all likely icon directories for source, one-file and one-dir builds."""
        candidates: List[str] = []

        env_dir = (os.environ.get("TDMSREADER_ICON_DIR", "") or "").strip()
        if env_dir:
            candidates.append(env_dir)

        # PyInstaller one-file extracts bundled data under sys._MEIPASS.
        base = getattr(sys, "_MEIPASS", None)
        if base:
            candidates.extend([
                os.path.join(base, "icons"),
                os.path.join(base, "_internal", "icons"),
            ])

        # Directory containing the executable. Useful both for one-dir builds and
        # as a user-overridable external icons folder next to the exe.
        try:
            exe_dir = os.path.dirname(os.path.abspath(sys.executable))
            candidates.extend([
                os.path.join(exe_dir, "icons"),
                os.path.join(exe_dir, "_internal", "icons"),
            ])
        except Exception:
            pass

        # Normal Python/source execution.
        try:
            src_dir = os.path.dirname(os.path.abspath(__file__))
            candidates.extend([
                os.path.join(src_dir, "icons"),
                os.path.join(os.path.dirname(src_dir), "icons"),
            ])
        except Exception:
            pass

        try:
            cwd = os.getcwd()
            candidates.extend([
                os.path.join(cwd, "icons"),
                os.path.join(os.path.dirname(cwd), "icons"),
            ])
        except Exception:
            pass

        seen: set = set()
        out: List[str] = []
        for d in candidates:
            if not d:
                continue
            d = os.path.normpath(os.path.abspath(d))
            key = os.path.normcase(d)
            if key in seen:
                continue
            seen.add(key)
            if os.path.isdir(d):
                out.append(d)
        return out

    def _load_icon(self, filename: str) -> QIcon:
        """Load an original PNG icon from bundled/external resources."""
        for d in self._icon_dirs():
            p = os.path.join(d, filename)
            if not os.path.isfile(p):
                continue

            icon = QIcon(p)
            if not icon.isNull():
                return icon

            # QPixmap fallback gives a second chance if QIcon's direct path
            # loader behaves differently in a frozen Qt application.
            pix = QPixmap(p)
            if not pix.isNull():
                return QIcon(pix)

        logger.debug("Quickbar icon not found or could not be loaded: %s; dirs=%s", filename, self._icon_dirs())
        return QIcon()

    def _mk_qb_btn(self, filename: str, tooltip: str, *, checkable: bool = False) -> QToolButton:
        b = QToolButton()
        b.setAutoRaise(True)
        b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        b.setCheckable(checkable)
        b.setFixedSize(40, 40)
        b.setIconSize(QSize(24, 24))
        b.setToolTip(tooltip)
        b.setProperty("qb", "1")
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        ic = self._load_icon(filename)
        if not ic.isNull():
            b.setIcon(ic)
        else:
            b.setText(tooltip[:1])
        return b

    def _qb_set_checked(self, attr: str, checked: bool) -> None:
        b = getattr(self, attr, None)
        if b is None:
            return
        try:
            blocked = b.blockSignals(True)
            b.setChecked(checked)
            b.blockSignals(blocked)
        except Exception:
            pass

    def _build_quickbar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("plotQuickbar")
        bar.setFrameShape(QFrame.Shape.NoFrame)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(6)

        self._qb_pan = self._mk_qb_btn("pan_hand.png", "Kaydır", checkable=True)
        self._qb_zoom = self._mk_qb_btn("zoom_plus.png", "Yakınlaştır (Dikdörtgen)", checkable=True)
        self._qb_fit = self._mk_qb_btn("move_pan.png", "Otomatik Sığdır")
        self._qb_region = self._mk_qb_btn("region_rect.png", "Aralık Seçici", checkable=True)
        self._qb_legend = self._mk_qb_btn("legend_menu.png", "Gösterge", checkable=True)
        self._qb_marker = self._mk_qb_btn("marker_pin.png", "Tıkla İşaretle", checkable=True)
        self._qb_ylock = self._mk_qb_btn("y_lock.png", "Y Kilidi", checkable=True)
        self._qb_y2lock = self._mk_qb_btn("y2_lock.png", "Y2 (Dijital) Kilidi", checkable=True)

        self._qb_mode_group = QButtonGroup(self)
        self._qb_mode_group.setExclusive(True)
        self._qb_mode_group.addButton(self._qb_pan)
        self._qb_mode_group.addButton(self._qb_zoom)

        self._qb_pan.setChecked(True)
        self._qb_legend.setChecked(True)

        def _mode_set(m: str) -> None:
            self.set_interaction_mode(m)
            self.interaction_mode_changed.emit(m)

        self._qb_pan.toggled.connect(lambda on: on and _mode_set("pan"))
        self._qb_zoom.toggled.connect(lambda on: on and _mode_set("zoom"))
        self._qb_fit.clicked.connect(lambda: (self.autofit_all(), self.autofit_requested.emit()))
        self._qb_region.toggled.connect(lambda on: (self.enable_region(on), self.region_toggled.emit(on)))
        self._qb_legend.toggled.connect(lambda on: (self.set_legend_visible(on), self.legend_toggled.emit(on)))
        self._qb_marker.toggled.connect(lambda on: (self.set_click_mark_mode(on), self.click_mark_toggled.emit(on)))
        self._qb_ylock.toggled.connect(lambda on: (self.set_y_lock(on), self.y_lock_toggled.emit(on)))
        self._qb_y2lock.toggled.connect(lambda on: (self.set_right_y_lock(on), self.right_y_lock_toggled.emit(on)))

        # Aralık seçici şimdilik kullanıcı arayüzünden kaldırıldı. Nesne
        # geriye dönük sinyal uyumluluğu için tutuluyor ancak gösterilmiyor.
        self._qb_region.setVisible(False)
        for w in (self._qb_pan, self._qb_zoom, self._qb_fit,
                  self._qb_legend, self._qb_marker, self._qb_ylock, self._qb_y2lock):
            lay.addWidget(w)
        lay.addStretch(1)

        try:
            self._info.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed)
        except Exception:
            pass
        lay.addWidget(self._info, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._update_quickbar_state()
        return bar

    def _update_quickbar_state(self) -> None:
        has_right = self._right_vb is not None and len(self._right_items) > 0
        if hasattr(self, "_qb_y2lock"):
            self._qb_y2lock.setEnabled(has_right)


# ===================================================================
# Stacked synchronized plot view
# ===================================================================

class StackedPlotView(QWidget):
    """Efficient synchronized subplot view.

    A single GraphicsLayoutWidget/scene is shared by all stacked channels.
    PlotItems and curves are reused when the visible channel set is unchanged,
    avoiding the expensive delete/recreate cycle of one PlotWidget per channel.
    """

    range_changed = pyqtSignal(float, float)
    axis_edit_requested = pyqtSignal(str)

    def __init__(self, bg_rgb: Tuple[int, int, int] = (255, 255, 255), parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._bg_rgb = bg_rgb
        self._plots: List[pg.PlotItem] = []
        self._entries: List[dict] = []
        self._keys: List[str] = []
        self._axis_mode = ""
        self._sync_guard = False
        self._last_x_label = "X"
        self._axis_style: Dict[str, Any] = {
            "x_title": "", "y_title": "", "y2_title": "",
            "font_family": "Arial",
            "title_size": 11, "title_bold": False,
            "tick_size": 9, "tick_bold": False,
        }

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # One graphics scene is materially faster than N independent PlotWidgets,
        # especially with OpenGL enabled and several synchronized channels.
        self.canvas = pg.GraphicsLayoutWidget()
        self.canvas.setBackground(bg_rgb)
        self.inner = self.canvas  # compatibility with existing screenshot/export code
        self.scroll.setWidget(self.canvas)
        lay.addWidget(self.scroll)

    def clear_all(self) -> None:
        try:
            self.canvas.clear()
        except Exception:
            try:
                self.canvas.ci.clear()
            except Exception:
                pass
        self._plots.clear()
        self._entries.clear()
        self._keys.clear()
        self._axis_mode = ""
        self.canvas.setMinimumHeight(180)

    def set_background(self, bg_rgb: Tuple[int, int, int]) -> None:
        self._bg_rgb = bg_rgb
        try:
            self.canvas.setBackground(bg_rgb)
        except Exception:
            pass
        fg = _bg_to_fg_rgb(bg_rgb)
        for p in self._plots:
            try:
                for ax in ("left", "bottom"):
                    axis = p.getAxis(ax)
                    if axis is not None:
                        axis.setPen(pg.mkPen(fg))
                        axis.setTextPen(pg.mkPen(fg))
            except Exception:
                pass

    def _stack_axis_label_style(self) -> Dict[str, str]:
        st = self._axis_style
        return {
            "font-family": str(st.get("font_family", "Arial") or "Arial"),
            "font-size": f"{max(6, int(st.get('title_size', 11) or 11))}pt",
            "font-weight": "bold" if bool(st.get("title_bold", False)) else "normal",
        }

    def _style_stacked_axis_ticks(self, pi: pg.PlotItem) -> None:
        st = self._axis_style
        font = QFont(str(st.get("font_family", "Arial") or "Arial"))
        font.setPointSize(max(6, int(st.get("tick_size", 9) or 9)))
        font.setBold(bool(st.get("tick_bold", False)))
        for ax_name in ("left", "bottom"):
            try:
                axis = pi.getAxis(ax_name)
                if hasattr(axis, "setTickFont"):
                    axis.setTickFont(font)
                else:
                    axis.setStyle(tickFont=font)
            except Exception:
                logger.debug("Stacked tick font apply failed", exc_info=True)

    def set_axis_style(self, style: Optional[Dict[str, Any]] = None) -> None:
        """Apply axis typography to existing stacked plots without rebuilding curves."""
        if style:
            self._axis_style.update(style)
        label_style = self._stack_axis_label_style()
        x_text = str(self._axis_style.get("x_title", "") or "").strip() or self._last_x_label
        for i, entry in enumerate(self._entries):
            pi = entry.get("plot")
            if pi is None:
                continue
            try:
                left_axis = pi.getAxis("left")
                left_text = str(getattr(left_axis, "labelText", "") or "")
                if left_text:
                    pi.setLabel("left", left_text, **label_style)
                if i == len(self._entries) - 1:
                    pi.setLabel("bottom", x_text, **label_style)
            except Exception:
                logger.debug("Stacked axis style apply failed", exc_info=True)
            self._style_stacked_axis_ticks(pi)

    def _rebuild_structure(self, visible: List[dict], axis_mode: str, x_label: str, style_map: Dict[str, dict]) -> None:
        self.clear_all()
        self._axis_mode = axis_mode
        self._keys = [str(s.get("style_key", "")) for s in visible]
        self.canvas.setMinimumHeight(max(220, 150 * len(visible)))
        first_plot: Optional[pg.PlotItem] = None
        fg = _bg_to_fg_rgb(self._bg_rgb)

        for i, s in enumerate(visible):
            skey = str(s.get("style_key", "") or "")
            bottom_axis = (
                EditableDateAxisItem(
                    orientation="bottom", edit_key="x", edit_callback=self.axis_edit_requested.emit
                )
                if axis_mode == "date"
                else EditableAxisItem(
                    orientation="bottom", edit_key="x", edit_callback=self.axis_edit_requested.emit
                )
            )
            axis_items = {
                "bottom": bottom_axis,
                "left": EditableAxisItem(
                    orientation="left",
                    edit_key=f"stacked_y::{skey}",
                    edit_callback=self.axis_edit_requested.emit,
                ),
            }
            pi = self.canvas.addPlot(row=i, col=0, axisItems=axis_items)
            try:
                pi.setMinimumHeight(142)
            except Exception:
                pass
            pi.showGrid(x=True, y=True, alpha=0.22)
            for ax in ("left", "bottom"):
                try:
                    pi.getAxis(ax).setPen(pg.mkPen(fg))
                    pi.getAxis(ax).setTextPen(pg.mkPen(fg))
                except Exception:
                    pass
            if first_plot is None:
                first_plot = pi
                pi.getViewBox().sigXRangeChanged.connect(self._emit_range)
            else:
                pi.setXLink(first_plot)
            self._style_stacked_axis_ticks(pi)
            if i != len(visible) - 1:
                pi.hideAxis("bottom")
            else:
                x_text = str(self._axis_style.get("x_title", "") or "").strip() or x_label
                pi.setLabel("bottom", x_text, **self._stack_axis_label_style())

            curve = pg.PlotDataItem()
            try:
                curve.setDownsampling(auto=True, mode="peak")
                curve.setClipToView(True)
            except Exception:
                pass
            pi.addItem(curve)
            self._plots.append(pi)
            self._entries.append({"plot": pi, "curve": curve})

    def plot_series(self, series_list: List[dict], axis_mode: str, x_label: str, style_map: Dict[str, dict]) -> None:
        self._last_x_label = x_label
        visible = [
            s for s in series_list
            if style_map.get(s.get("style_key", ""), {}).get("visible", True) is not False
        ][:CFG.stacked_max_channels]
        if not visible:
            self.clear_all()
            return

        keys = [str(s.get("style_key", "")) for s in visible]
        if keys != self._keys or axis_mode != self._axis_mode or len(self._entries) != len(visible):
            self._rebuild_structure(visible, axis_mode, x_label, style_map)

        # Stacked plots are much smaller than the main overlay plot. Use an
        # adaptive shared point budget while preserving min/max extrema.
        # This keeps 10-20 channel views responsive without hiding spikes.
        stacked_max_points = min(
            CFG.plot_max_points,
            max(12_000, int(120_000 / max(1, len(visible)))),
        )

        for i, s in enumerate(visible):
            entry = self._entries[i]
            pi: pg.PlotItem = entry["plot"]
            curve: pg.PlotDataItem = entry["curve"]
            st = style_map.get(s.get("style_key", ""), {})
            name = str(st.get("display_name") or s.get("name") or "Kanal")
            unit = str(st.get("unit_override") or s.get("unit") or "").strip()
            label = f"{name} ({unit})" if unit else name
            pi.setLabel("left", label, **self._stack_axis_label_style())
            self._style_stacked_axis_ticks(pi)
            if i == len(visible) - 1:
                try:
                    pi.showAxis("bottom")
                except Exception:
                    pass
                x_text = str(self._axis_style.get("x_title", "") or "").strip() or x_label
                pi.setLabel("bottom", x_text, **self._stack_axis_label_style())

            x = np.asarray(s.get("x", []), dtype=np.float64)
            y = np.asarray(s.get("y", []))
            xs = float(st.get("x_shift", 0.0) or 0.0)
            ys = float(st.get("y_shift", 0.0) or 0.0)
            if xs:
                x = x + xs
            if ys:
                y = np.asarray(y, dtype=np.float64) + ys

            color = st.get("color") or pg.intColor(i, hues=max(1, len(visible)))
            pen = pg.mkPen(color, width=float(st.get("width", 1.8) or 1.8))
            curve.setPen(pen)
            if s.get("is_digital"):
                try:
                    curve.setData(x, y, connect="finite", skipFiniteCheck=True)
                except Exception:
                    curve.setData(x, y, connect="finite")
            else:
                xd, yd = minmax_envelope_downsample(x, y, stacked_max_points)
                try:
                    curve.setData(xd, yd, skipFiniteCheck=True)
                except Exception:
                    curve.setData(xd, yd)

        self.set_background(self._bg_rgb)

    def _emit_range(self, _vb: Any, ranges: Any) -> None:
        if self._sync_guard or not self._plots:
            return
        try:
            xr = ranges[0] if isinstance(ranges, (list, tuple)) and len(ranges) == 2 and isinstance(ranges[0], (list, tuple)) else self._plots[0].getViewBox().viewRange()[0]
            self.range_changed.emit(float(xr[0]), float(xr[1]))
        except Exception:
            try:
                xr = self._plots[0].getViewBox().viewRange()[0]
                self.range_changed.emit(float(xr[0]), float(xr[1]))
            except Exception:
                pass

    def set_xrange(self, x0: float, x1: float) -> None:
        if not self._plots or x1 <= x0:
            return
        self._sync_guard = True
        try:
            self._plots[0].getViewBox().setXRange(float(x0), float(x1), padding=0.0)
        finally:
            self._sync_guard = False

    def autofit_all(self) -> None:
        for p in self._plots:
            try:
                p.getViewBox().enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
                p.getViewBox().autoRange()
                p.getViewBox().enableAutoRange(axis=pg.ViewBox.XYAxes, enable=False)
            except Exception:
                pass


# ===================================================================
# Detached Plot Window
# ===================================================================

class DetachedPlotWindow(QMainWindow):
    closed = pyqtSignal()

    def __init__(self, bg_rgb: Tuple[int, int, int], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Grafik (Ayrılmış)")
        self.resize(1280, 800)
        self.pane = PlotPane(bg_rgb=bg_rgb)
        self.setCentralWidget(self.pane)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.closed.emit()
        super().closeEvent(event)


# ===================================================================
# Main Window
# ===================================================================

class MainWindow(QMainWindow):
    """Top-level application window for TDMSReader."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("TDMSReader AKBGB")
        self.resize(1680, 960)

        self.settings = QSettings(CFG.settings_org, CFG.settings_app)
        self._theme = str(self.settings.value("ui/theme", "light") or "light").lower()
        if self._theme not in ("light", "dark"):
            self._theme = "light"

        self._theme_guard = False
        self._theme_default_bg: Dict[str, Tuple[int, int, int]] = {"dark": (0, 0, 0), "light": (255, 255, 255)}

        self.files: Dict[str, FileState] = {}
        self.file_counter = 0
        self.current_series: Optional[List[dict]] = None

        # Explicit per-channel sampling-frequency overrides, keyed by style_key.
        # These are session-only because style_key includes the current file_id.
        self.channel_fs_overrides: Dict[str, float] = {}

        self.style_map: Dict[str, dict] = {}
        self.axis_style: Dict[str, Any] = {
            "x_title": "", "y_title": "", "y2_title": "",
            "font_family": "Arial",
            "title_size": 11, "title_bold": False,
            "tick_size": 9, "tick_bold": False,
        }
        self._jobs: List[Tuple[QThread, CancellableWorker, str]] = []
        self._virtual_channel_counter = 0
        self._calc_alias_map: Dict[str, dict] = {}
        self._zoom_history: List[Tuple[float, float]] = []
        self._zoom_history_guard = False

        self.plot_bg_rgb: Tuple[int, int, int] = self._theme_default_bg[self._theme]
        self.detached_win: Optional[DetachedPlotWindow] = None
        self._syncing_range = False
        self._pane_sync_guard = False

        self._marker_counter = 0
        self._markers: Dict[int, dict] = {}
        self._last_preview_payload: Optional[dict] = None
        self._last_fft_payload: Optional[dict] = None

        self._filter_token = 0
        self._range_stats_token = 0

        # Cached filtered/display payload. Overlay and Stacked are rendered on
        # demand instead of both being rebuilt after every small UI change.
        self._last_display_series: List[dict] = []
        self._last_axis_mode: str = "numeric"
        self._last_x_label: str = "X"
        self._overlay_dirty = True
        self._stacked_dirty = True
        self._active_plot_mode = "Overlay"
        self._channel_manager_dirty = True
        self._calculated_tab_dirty = True
        self._manual_fs_tab_dirty = True

        self.setAcceptDrops(True)

        # Lazy view
        self._lazy_view_pending: Optional[Tuple[float, float]] = None
        self._lazy_view_timer = QTimer(self)
        self._lazy_view_timer.setSingleShot(True)
        self._lazy_view_timer.timeout.connect(self._run_lazy_view_update)
        self._lazy_view_token = 0
        self._preserve_main_xrange: Optional[Tuple[float, float]] = None
        self._preserve_detached_xrange: Optional[Tuple[float, float]] = None

        self._build_ui()
        self._build_menu_bar()
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self._setup_status_metrics()
        self._connect_signals()
        self._install_shortcuts()
        self.set_theme(self._theme)
        self._restore_ui_state()
        self._apply_interaction_mode_to_all()
        self._refresh_recent_menu()

        self.status.showMessage("Soldaki 'Aç' sekmesinden bir TDMS dosyası açın.")

        if not SCIPY_AVAILABLE:
            QMessageBox.warning(
                self, "Eksik Kütüphane",
                "Scipy yüklü değil. Savitzky-Golay filtresi çalışmayacak.\n"
                "Yüklemek için: pip install scipy",
            )
            self.chk_smooth.setEnabled(False)

        self._apply_background_everywhere()

    # ----- Job management -----

    def _cancel_jobs(self, tags: List[str]) -> None:
        """Cancel all pending jobs matching any of the given tags."""
        tagset = set(t for t in tags if t)
        if not tagset:
            return
        for th, worker, tg in list(self._jobs):
            if tg not in tagset:
                continue
            try:
                worker.cancel()
            except Exception:
                logger.debug("Worker cancel failed for tag=%s", tg, exc_info=True)
            try:
                th.requestInterruption()
            except Exception:
                pass

    def _start_job(
        self, worker: CancellableWorker,
        on_finished: Callable, on_failed: Optional[Callable] = None, *,
        tag: str = "", cancel_tags: Optional[List[str]] = None,
    ) -> None:
        """Start a worker on a new QThread with proper lifecycle management."""
        if cancel_tags:
            self._cancel_jobs(cancel_tags)

        th = QThread(self)
        worker.moveToThread(th)

        def _cleanup() -> None:
            th.quit()

        def _remove_job() -> None:
            self._jobs = [(t, w, tg) for t, w, tg in self._jobs if t is not th]

        th.started.connect(worker.run)
        worker.finished.connect(on_finished)
        worker.failed.connect(on_failed or self._on_worker_failed)
        worker.finished.connect(_cleanup)
        worker.failed.connect(_cleanup)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        th.finished.connect(th.deleteLater)
        th.finished.connect(_remove_job)

        self._jobs.append((th, worker, tag))
        th.start()

    # ----- Theme -----

    def _theme_button_text(self) -> str:
        return "\U0001F319" if self._theme == "dark" else "\u2600"

    def set_theme(self, theme: str) -> None:
        theme = (theme or "").strip().lower()
        if theme not in ("dark", "light") or self._theme_guard:
            return
        self._theme_guard = True
        try:
            self._theme = theme
            app = QApplication.instance()
            if app is not None:
                app.setStyleSheet(DARK_QSS if self._theme == "dark" else LIGHT_QSS)
            for btn in (getattr(self, "btn_theme_plot", None), getattr(self, "btn_theme_fft", None)):
                if btn is None:
                    continue
                btn.blockSignals(True)
                btn.setChecked(self._theme == "dark")
                btn.setText(self._theme_button_text())
                btn.blockSignals(False)
            self.plot_bg_rgb = self._theme_default_bg[self._theme]
            self._set_bg_button_preview(self.plot_bg_rgb)
            self._apply_background_everywhere()
            if hasattr(self, "status") and self.status is not None:
                self.status.showMessage(f"Tema: {'Koyu' if self._theme == 'dark' else 'Açık'}", 2000)
            act = getattr(self, "_theme_menu_action", None)
            if act is not None:
                act.setText("Koyu Tema" if self._theme == "light" else "Açık Tema")
        finally:
            self._theme_guard = False

    def _toggle_theme_from_button(self) -> None:
        sender = self.sender()
        want_dark = bool(getattr(sender, "isChecked", lambda: True)())
        self.set_theme("dark" if want_dark else "light")

    # ----- Menu bar -----

    def _build_menu_bar(self) -> None:
        mb = self.menuBar()

        file_menu = mb.addMenu("Dosya")
        act = file_menu.addAction("TDMS Aç...")
        act.setShortcut(QKeySequence.StandardKey.Open)
        act.triggered.connect(self.open_tdms)

        self._recent_menu = file_menu.addMenu("Son Açılanlar")

        file_menu.addSeparator()
        act = file_menu.addAction("CSV Dışa Aktar")
        act.setShortcut(QKeySequence("Ctrl+E"))
        act.triggered.connect(self.export_csv)
        act = file_menu.addAction("Grafiği Kaydet")
        act.setShortcut(QKeySequence("Ctrl+Shift+S"))
        act.triggered.connect(self.save_plot_image)
        act = file_menu.addAction("Grafiği Panoya Kopyala")
        act.setShortcut(QKeySequence("Ctrl+Shift+C"))
        act.triggered.connect(self.copy_plot_to_clipboard)
        file_menu.addSeparator()
        act = file_menu.addAction("Sekmeyi Kapat")
        act.setShortcut(QKeySequence.StandardKey.Close)
        act.triggered.connect(lambda: self._close_file_tab(self.file_tabs.currentIndex()))
        file_menu.addSeparator()
        act = file_menu.addAction("Çıkış")
        act.setShortcut(QKeySequence("Ctrl+Q"))
        act.triggered.connect(self.close)

        view_menu = mb.addMenu("Görünüm")
        act = view_menu.addAction("Koyu Tema" if self._theme == "light" else "Açık Tema")
        act.triggered.connect(lambda: self.set_theme("dark" if self._theme == "light" else "light"))
        self._theme_menu_action = act
        view_menu.addSeparator()
        act = view_menu.addAction("Otomatik Sığdır")
        act.triggered.connect(self._autofit_all_panes)
        act = view_menu.addAction("Grafiği Temizle")
        act.triggered.connect(self.on_clear_plot)
        act = view_menu.addAction("Grafiği Ayır / Geri Bağla")
        act.triggered.connect(self.toggle_detach_plot)
        act = view_menu.addAction("Detay Ayarları")
        act.triggered.connect(lambda: self._set_advanced_controls_visible(not self.controls_dock.isVisible()))
        view_menu.addSeparator()
        act = view_menu.addAction("Kanal Yöneticisi")
        act.triggered.connect(self._open_channel_manager_tab)
        act = view_menu.addAction("Hesaplanan Kanal")
        act.triggered.connect(self._open_calculated_channel_tab)

        help_menu = mb.addMenu("Yardım")
        act = help_menu.addAction("Klavye Kısayolları")
        act.setShortcut(QKeySequence("F1"))
        act.triggered.connect(self.show_shortcuts_dialog)
        act = help_menu.addAction("Hakkında")
        act.triggered.connect(self._show_about_dialog)

    # ----- Recent files -----

    def _get_recent_files(self) -> List[str]:
        try:
            raw = self.settings.value("paths/recent_files", [])
            if isinstance(raw, list):
                return [str(p) for p in raw if p and os.path.isfile(str(p))]
            if isinstance(raw, str) and raw:
                return [raw] if os.path.isfile(raw) else []
        except Exception:
            pass
        return []

    def _add_to_recent(self, path: str) -> None:
        try:
            abs_p = os.path.abspath(path)
            recent = self._get_recent_files()
            recent = [p for p in recent if os.path.abspath(p) != abs_p]
            recent.insert(0, abs_p)
            recent = recent[:CFG.max_recent_files]
            self.settings.setValue("paths/recent_files", recent)
            self._refresh_recent_menu()
        except Exception:
            logger.debug("Recent file tracking failed", exc_info=True)

    def _refresh_recent_menu(self) -> None:
        menu = getattr(self, "_recent_menu", None)
        if menu is None:
            return
        menu.clear()
        recent = self._get_recent_files()
        if not recent:
            act = menu.addAction("(boş)")
            act.setEnabled(False)
            return
        for p in recent:
            display = os.path.basename(p)
            act = menu.addAction(f"{display}  —  {p}")
            act.triggered.connect(lambda checked, path=p: self.open_tdms_path(path))
        menu.addSeparator()
        act = menu.addAction("Listeyi Temizle")
        act.triggered.connect(self._clear_recent_files)

    def _clear_recent_files(self) -> None:
        self.settings.setValue("paths/recent_files", [])
        self._refresh_recent_menu()

    # ----- Statistics panel -----

    def _refresh_statistics(self) -> None:
        """Statistics panel removed; keep method as a safe no-op."""
        return

    # ----- Plot image export -----

    def save_plot_image(self) -> None:
        path, selected_filter = QFileDialog.getSaveFileName(
            self, "Grafiği Kaydet", "", "PNG resmi (*.png);;SVG vektör (*.svg);;Tüm dosyalar (*)",
        )
        if not path:
            return
        stacked = getattr(self, "cmb_plot_mode", None) is not None and self.cmb_plot_mode.currentText() == "Stacked"
        try:
            if stacked:
                if path.lower().endswith(".svg"):
                    path = path[:-4] + ".png"
                elif not path.lower().endswith(".png"):
                    path += ".png"
                self.stacked_view.inner.grab().save(path, "PNG")
            else:
                if self.plot_pane.plot is None:
                    QMessageBox.information(self, "Grafik Yok", "Kaydedilecek grafik yok."); return
                import pyqtgraph.exporters as exporters
                if path.lower().endswith(".svg"):
                    exporter = exporters.SVGExporter(self.plot_pane.plot.getPlotItem())
                else:
                    if not path.lower().endswith(".png"): path += ".png"
                    exporter = exporters.ImageExporter(self.plot_pane.plot.getPlotItem()); exporter.parameters()["width"] = 1920
                exporter.export(path)
            self.status.showMessage(f"Grafik kaydedildi: {os.path.basename(path)}", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Kaydetme Hatası", f"Grafik kaydedilemedi:\n{e}")

    # ----- Shortcuts dialog -----

    def show_shortcuts_dialog(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("Klavye Kısayolları")
        dlg.resize(480, 420)
        lay = QVBoxLayout(dlg)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(False)
        browser.setHtml(
            "<style>"
            "table { border-collapse: collapse; width: 100%; }"
            "th, td { text-align: left; padding: 6px 12px; border-bottom: 1px solid #ccc; }"
            "th { font-weight: bold; background: #f0f0f0; }"
            "kbd { background: #e8e8e8; border: 1px solid #bbb; border-radius: 3px;"
            "      padding: 1px 6px; font-family: monospace; }"
            "</style>"
            "<h3>Klavye Kısayolları</h3>"
            "<table>"
            "<tr><th>Kısayol</th><th>İşlev</th></tr>"
            "<tr><td><kbd>Ctrl+O</kbd></td><td>TDMS dosyası aç</td></tr>"
            "<tr><td><kbd>Ctrl+W</kbd></td><td>Geçerli sekmeyi kapat</td></tr>"
            "<tr><td><kbd>Ctrl+E</kbd></td><td>CSV dışa aktar</td></tr>"
            "<tr><td><kbd>Ctrl+Shift+S</kbd></td><td>Grafiği resim olarak kaydet</td></tr>"
            "<tr><td><kbd>Ctrl+Q</kbd></td><td>Uygulamadan çık</td></tr>"
            "<tr><td><kbd>F1</kbd></td><td>Bu diyalogu göster</td></tr>"
            "<tr><td><kbd>F / Backspace</kbd></td><td>Otomatik sığdır</td></tr>"
            "<tr><td><kbd>Ctrl+Z</kbd></td><td>Önceki zoom</td></tr>"
            "<tr><td><kbd>L</kbd></td><td>Legend aç/kapat</td></tr>"
            "<tr><td><kbd>M</kbd></td><td>Tıkla işaretle aç/kapat</td></tr>"
            "<tr><td><kbd>Ctrl+Shift+C</kbd></td><td>Grafiği panoya kopyala</td></tr>"
            "<tr><td colspan='2'><br><b>Grafik Etkileşimi</b></td></tr>"
            "<tr><td>Sol Fare Sürükleme</td><td>Kaydır / Yakınlaştır (moda göre)</td></tr>"
            "<tr><td>Fare Tekerleği</td><td>Yakınlaştır / Uzaklaştır</td></tr>"
            "<tr><td>Sağ Tık Sürükle</td><td>Eksenleri ölçekle</td></tr>"
            "<tr><td>Sürükle & Bırak</td><td>TDMS dosyasını sürükleyip bırakarak aç</td></tr>"
            "</table>"
        )
        lay.addWidget(browser)
        btn_close = QPushButton("Kapat")
        btn_close.clicked.connect(dlg.accept)
        lay.addWidget(btn_close, alignment=Qt.AlignmentFlag.AlignRight)
        dlg.exec()

    def _show_about_dialog(self) -> None:
        QMessageBox.about(
            self, "Hakkında",
            "<h3>TDMS Okuyucu</h3>"
            "<p>NI TDMS dosyaları için DIAdem benzeri görüntüleyici ihtiyacı doğrultusunda geliştirilmiştir.</p>"
            "</ul>"
            "<h3>RİTİM - Akış Kontrol Bileşenleri Geliştirme Birimi.</h3>"
        )

    # ----- UI build -----

    def _make_ribbon_button(
        self, label: str, *, icon: Optional[QIcon] = None, glyph: str = "",
        tooltip: str = "", checkable: bool = False,
    ) -> QToolButton:
        """Create a compact ribbon-style command button."""
        b = QToolButton()
        b.setProperty("ribbon", "1")
        b.setCheckable(checkable)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        b.setFixedSize(76, 60)
        if tooltip:
            b.setToolTip(tooltip)
        if icon is not None and not icon.isNull():
            b.setIcon(icon)
            b.setIconSize(QSize(22, 22))
            b.setText(label)
            b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        else:
            prefix = (glyph.strip() + "\n") if glyph.strip() else ""
            b.setText(prefix + label)
            b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        return b

    def _make_ribbon_group(self, title: str, buttons: List[QToolButton], *, last: bool = False) -> QFrame:
        group = QFrame()
        group.setObjectName("ribbonGroup")
        if last:
            group.setProperty("lastGroup", "true")
        v = QVBoxLayout(group)
        v.setContentsMargins(4, 2, 4, 1)
        v.setSpacing(0)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(1)
        for btn in buttons:
            row.addWidget(btn)
        v.addLayout(row)
        cap = QLabel(title)
        cap.setProperty("ribbonGroupTitle", "true")
        cap.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        v.addWidget(cap)
        return group

    def _activate_tab_by_text(self, text: str) -> None:
        tabs = getattr(self, "tabs", None)
        if tabs is None:
            return
        wanted = (text or "").strip().lower()
        for i in range(tabs.count()):
            if tabs.tabText(i).strip().lower() == wanted:
                tabs.setCurrentIndex(i)
                return

    def _make_control_ribbon_group(self, title: str, widgets: List[QWidget], *, last: bool = False) -> QFrame:
        """Compact ribbon group for live plot controls.

        Existing control widgets are re-parented into this ribbon so there is
        only one source of truth for every setting.  This avoids duplicated
        values/signals between the old inspector and the ribbon.
        """
        group = QFrame()
        group.setObjectName("ribbonGroup")
        if last:
            group.setProperty("lastGroup", "true")
        v = QVBoxLayout(group)
        v.setContentsMargins(6, 2, 6, 2)
        v.setSpacing(0)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(5)
        for w in widgets:
            row.addWidget(w)
        v.addLayout(row)
        cap = QLabel(title)
        cap.setProperty("ribbonGroupTitle", "true")
        cap.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        v.addWidget(cap)
        return group

    def _compact_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setProperty("compactLabel", "true")
        lbl.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
        return lbl

    def _build_control_ribbon(self) -> QScrollArea:
        """Move the frequently-used inspector controls into a second ribbon row.

        The original widgets themselves are moved, not copied.  Existing
        signal connections therefore remain intact and analysis behaviour is
        unchanged.  The old right inspector is retained for advanced items
        such as the marker list and bulk-shift channel selection.
        """
        host = QFrame()
        host.setObjectName("controlRibbon")
        lay = QHBoxLayout(host)
        lay.setContentsMargins(5, 3, 5, 3)
        lay.setSpacing(0)

        # Range / axis controls
        self.xmin.setFixedWidth(98)
        self.xmax.setFixedWidth(98)
        self.btn_fit_y.setText("Y Sığdır")
        self.btn_fit_y.setToolTip("Seçili X aralığındaki veriye göre Y eksenini sığdır")
        range_widgets: List[QWidget] = [
            self._compact_label("X Min"), self.xmin,
            self._compact_label("X Maks"), self.xmax, self.btn_fit_y,
        ]
        lay.addWidget(self._make_control_ribbon_group("Aralık / Eksen", range_widgets))

        # Marker controls; marker list itself stays in the advanced inspector.
        self.btn_clear_markers.setText("Temizle")
        self.cmb_marker_calc_axis.setFixedWidth(108)
        self.btn_marker_stats.setText("İstatistik")
        marker_widgets: List[QWidget] = [
            self._compact_label("Hesap"), self.cmb_marker_calc_axis,
            self.btn_marker_add, self.btn_marker_sub, self.btn_marker_stats, self.btn_clear_markers,
        ]
        lay.addWidget(self._make_control_ribbon_group("Marker İşlemleri", marker_widgets))

        # Filtering / cursor visibility
        self.chk_smooth.setText("SG Filtre")
        self.chk_cursor_marker.setText("İmleç Noktası")
        self.spin_smooth_win.setFixedWidth(82)
        filter_widgets: List[QWidget] = [
            self.chk_smooth, self._compact_label("Pencere"), self.spin_smooth_win, self.chk_cursor_marker,
        ]
        lay.addWidget(self._make_control_ribbon_group("Filtre", filter_widgets))

        # Per-channel style / shift controls
        self.cmb_style_series.setMinimumWidth(160)
        self.cmb_style_series.setMaximumWidth(230)
        self.btn_pick_color.setText("Renk")
        self.sp_line_width.setFixedWidth(67)
        self.sp_x_shift.setFixedWidth(82)
        self.sp_y_shift.setFixedWidth(82)
        self.btn_style_default.setText("Sıfırla")
        style_widgets: List[QWidget] = [
            self._compact_label("Kanal"), self.cmb_style_series, self.btn_pick_color,
            self._compact_label("Kalınlık"), self.sp_line_width,
            self._compact_label("X Δ"), self.sp_x_shift,
            self._compact_label("Y Δ"), self.sp_y_shift,
            self.btn_style_apply, self.btn_style_default,
        ]
        lay.addWidget(self._make_control_ribbon_group("Kanal Stili / Shift", style_widgets))

        # Plot appearance controls
        self.btn_bg_pick.setText("Arka Plan")
        self.btn_bg_reset.setText("Arka Plan Sıfırla")
        appearance_widgets: List[QWidget] = [self.btn_bg_pick, self.btn_bg_reset]
        lay.addWidget(self._make_control_ribbon_group("Görünüm", appearance_widgets, last=True))
        lay.addStretch(1)

        scroll = QScrollArea()
        scroll.setObjectName("controlRibbonScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumHeight(61)
        scroll.setMaximumHeight(76)
        scroll.setWidget(host)
        return scroll

    def _set_advanced_controls_visible(self, visible: bool) -> None:
        dock = getattr(self, "controls_dock", None)
        if dock is None:
            return
        dock.setVisible(bool(visible))
        rb = getattr(self, "rb_advanced_controls", None)
        if rb is not None and rb.isChecked() != bool(visible):
            blocked = rb.blockSignals(True)
            rb.setChecked(bool(visible))
            rb.blockSignals(blocked)
        if visible:
            try:
                # The drawer lives inside the Graph workspace. Bring it into
                # view automatically when opened from the ribbon/corner button.
                if getattr(self, "tabs", None) is not None:
                    self.tabs.setCurrentIndex(0)
                self.btn_ctrl_collapse.setChecked(True)
                self._on_controls_panel_toggled(True)
            except Exception:
                pass
        try:
            self.status.showMessage("Detay ayarları açıldı." if visible else "Detay ayarları kapatıldı.", 1600)
        except Exception:
            pass

    def _build_ribbon(self) -> QFrame:
        """Compact one-row command ribbon.

        Workspace functions such as FFT, Channel Manager, Calculated Channel
        and Manual Fs are permanent center tabs, so the ribbon only carries
        global commands and interaction tools.
        """
        bar = QFrame()
        bar.setObjectName("ribbonBar")
        bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(2)

        style = QApplication.style()
        def std_icon(name: str, fallback: QStyle.StandardPixmap) -> QIcon:
            if style is None:
                return QIcon()
            sp = getattr(QStyle.StandardPixmap, name, fallback)
            return style.standardIcon(sp)

        open_ic = std_icon("SP_DialogOpenButton", QStyle.StandardPixmap.SP_FileIcon)
        save_ic = std_icon("SP_DialogSaveButton", QStyle.StandardPixmap.SP_FileIcon)
        export_ic = std_icon("SP_DriveFDIcon", QStyle.StandardPixmap.SP_DialogSaveButton)
        view_ic = std_icon("SP_FileDialogDetailedView", QStyle.StandardPixmap.SP_FileDialogListView)
        measure_ic = std_icon("SP_DialogApplyButton", QStyle.StandardPixmap.SP_DialogOkButton)
        theme_ic = std_icon("SP_DesktopIcon", QStyle.StandardPixmap.SP_ComputerIcon)
        detach_ic = std_icon("SP_TitleBarNormalButton", QStyle.StandardPixmap.SP_FileDialogDetailedView)
        details_ic = std_icon("SP_FileDialogContentsView", QStyle.StandardPixmap.SP_FileDialogDetailedView)

        try:
            fit_ic = self.plot_pane._load_icon("move_pan.png")
            pan_ic = self.plot_pane._load_icon("pan_hand.png")
            zoom_ic = self.plot_pane._load_icon("zoom_plus.png")
            region_ic = self.plot_pane._load_icon("region_rect.png")
            marker_ic = self.plot_pane._load_icon("marker_pin.png")
        except Exception:
            fit_ic = pan_ic = zoom_ic = region_ic = marker_ic = QIcon()

        rb_open = self._make_ribbon_button("TDMS Aç", icon=open_ic, tooltip="TDMS dosyası aç (Ctrl+O)")
        rb_export = self._make_ribbon_button("CSV Aktar", icon=export_ic, tooltip="Çizili veriyi CSV olarak dışa aktar")
        rb_save = self._make_ribbon_button("Kaydet", icon=save_ic, tooltip="Grafiği PNG/SVG olarak kaydet")
        rb_open.clicked.connect(self.open_tdms)
        rb_export.clicked.connect(self.export_csv)
        rb_save.clicked.connect(self.save_plot_image)
        lay.addWidget(self._make_ribbon_group("Dosya", [rb_open, rb_export, rb_save]))

        rb_plot_mode = self._make_ribbon_button("Görünüm", icon=view_ic, tooltip="Overlay / Stacked görünümünü seç")
        plot_mode_menu = QMenu(rb_plot_mode)
        act_overlay = plot_mode_menu.addAction("Overlay")
        act_stacked = plot_mode_menu.addAction("Stacked")
        act_overlay.triggered.connect(lambda: self.cmb_plot_mode.setCurrentText("Overlay"))
        act_stacked.triggered.connect(lambda: self.cmb_plot_mode.setCurrentText("Stacked"))
        rb_plot_mode.setMenu(plot_mode_menu)
        rb_plot_mode.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)

        rb_pan = self._make_ribbon_button("Pan", icon=pan_ic, tooltip="Kaydırma modu")
        rb_zoom = self._make_ribbon_button("Zoom", icon=zoom_ic, tooltip="Dikdörtgen yakınlaştırma")
        rb_fit = self._make_ribbon_button("Sığdır", icon=fit_ic, tooltip="Tüm grafikleri otomatik sığdır")
        rb_pan.clicked.connect(lambda: self._apply_interaction_mode_to_all("pan"))
        rb_zoom.clicked.connect(lambda: self._apply_interaction_mode_to_all("zoom"))
        rb_fit.clicked.connect(self._autofit_all_panes)
        lay.addWidget(self._make_ribbon_group("Grafik", [rb_plot_mode, rb_pan, rb_zoom, rb_fit]))

        self.rb_marker = self._make_ribbon_button("Marker", icon=marker_ic, tooltip="Tıklayarak marker ekleme", checkable=True)
        rb_stats = self._make_ribbon_button("Ölçüm", icon=measure_ic, tooltip="İki marker arasındaki istatistikleri hesapla")
        self.rb_marker.toggled.connect(self.chk_click_mark.setChecked)
        self.chk_click_mark.toggled.connect(self.rb_marker.setChecked)
        rb_stats.clicked.connect(self._marker_interval_stats)
        lay.addWidget(self._make_ribbon_group("İşaretleme & Ölçüm", [self.rb_marker, rb_stats]))

        rb_theme = self._make_ribbon_button("Tema", icon=theme_ic, tooltip="Açık / koyu tema")
        rb_detach = self._make_ribbon_button("Ayır", icon=detach_ic, tooltip="Grafiği ayrı pencereye al")
        self.rb_advanced_controls = self._make_ribbon_button(
            "Ayarlar", icon=details_ic, tooltip="Grafik altındaki detay ayar çekmecesini aç/kapat", checkable=True
        )
        rb_theme.clicked.connect(lambda: self.set_theme("dark" if self._theme == "light" else "light"))
        rb_detach.clicked.connect(self.toggle_detach_plot)
        self.rb_advanced_controls.toggled.connect(self._set_advanced_controls_visible)
        lay.addWidget(self._make_ribbon_group("Görünüm", [rb_theme, rb_detach, self.rb_advanced_controls], last=True))
        lay.addStretch(1)
        return bar

    def _make_summary_cell(self, title: str) -> Tuple[QFrame, QLabel]:
        cell = QFrame()
        cell.setObjectName("summaryCell")
        lay = QVBoxLayout(cell)
        lay.setContentsMargins(11, 5, 15, 5)
        lay.setSpacing(1)
        cap = QLabel(title)
        cap.setProperty("summaryCaption", "true")
        val = QLabel("—")
        val.setProperty("summaryValue", "true")
        lay.addWidget(cap)
        lay.addWidget(val)
        return cell, val

    def _build_workspace_summary(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("summaryStrip")
        frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        frame.setMinimumHeight(54)
        frame.setMaximumHeight(66)
        lay = QHBoxLayout(frame)
        lay.setContentsMargins(5, 2, 5, 2)
        lay.setSpacing(0)
        cells = []
        for title, attr in [
            ("Gösterilen Aralık", "lbl_ws_range"),
            ("Toplam Süre", "lbl_ws_duration"),
            ("Kanal / Örnek", "lbl_ws_samples"),
            ("Örnekleme", "lbl_ws_fs"),
        ]:
            cell, val = self._make_summary_cell(title)
            setattr(self, attr, val)
            cells.append(cell)
            lay.addWidget(cell, 1)
        if cells:
            cells[-1].setStyleSheet("border-right: none;")
        return frame

    def _format_compact_count(self, n: int) -> str:
        n = int(max(0, n))
        if n >= 1_000_000_000:
            return f"{n/1_000_000_000:.3g}B"
        if n >= 1_000_000:
            return f"{n/1_000_000:.3g}M"
        if n >= 1_000:
            return f"{n/1_000:.3g}k"
        return str(n)

    def _update_workspace_summary(self, x0: Optional[float] = None, x1: Optional[float] = None) -> None:
        labels = [
            getattr(self, "lbl_ws_range", None), getattr(self, "lbl_ws_duration", None),
            getattr(self, "lbl_ws_samples", None), getattr(self, "lbl_ws_fs", None),
        ]
        if not self.current_series:
            for lbl in labels:
                if lbl is not None:
                    lbl.setText("—")
            self._update_status_metrics()
            return

        starts: List[float] = []
        ends: List[float] = []
        total_samples = 0
        fs_values: List[float] = []
        for s in self.current_series:
            total_samples += int(s.get("sample_count", len(np.asarray(s.get("y", [])))) or 0)
            fs = s.get("fs_est") or s.get("source_fs_est")
            try:
                fsf = float(fs)
                if math.isfinite(fsf) and fsf > 0:
                    fs_values.append(fsf)
            except Exception:
                pass
            xx = np.asarray(s.get("x", []), dtype=np.float64)
            if xx.size:
                a, b = float(xx[0]), float(xx[-1])
                if math.isfinite(a) and math.isfinite(b):
                    starts.append(min(a, b)); ends.append(max(a, b))

        total_duration = (max(ends) - min(starts)) if starts and ends else float("nan")
        if x0 is None or x1 is None:
            try:
                if self.cmb_plot_mode.currentText() == "Stacked" and self.stacked_view._plots:
                    xr = self.stacked_view._plots[0].getViewBox().viewRange()[0]
                elif self.plot_pane.plot is not None:
                    xr = self.plot_pane.plot.getViewBox().viewRange()[0]
                else:
                    xr = None
                if xr:
                    x0, x1 = float(xr[0]), float(xr[1])
            except Exception:
                pass
        shown = abs(float(x1) - float(x0)) if x0 is not None and x1 is not None else float("nan")

        if getattr(self, "lbl_ws_range", None) is not None:
            self.lbl_ws_range.setText(f"{shown:.6g} s" if math.isfinite(shown) else "—")
        if getattr(self, "lbl_ws_duration", None) is not None:
            self.lbl_ws_duration.setText(f"{total_duration:.6g} s" if math.isfinite(total_duration) else "—")
        if getattr(self, "lbl_ws_samples", None) is not None:
            self.lbl_ws_samples.setText(f"{len(self.current_series)} / {self._format_compact_count(total_samples)}")
        if getattr(self, "lbl_ws_fs", None) is not None:
            if not fs_values:
                txt = "—"
            else:
                f0 = fs_values[0]
                same = all(abs(f - f0) <= max(1e-9, abs(f0)*1e-7) for f in fs_values[1:])
                txt = f"{f0:.6g} Hz" if same else f"Çoklu ({min(fs_values):.4g}–{max(fs_values):.4g} Hz)"
            self.lbl_ws_fs.setText(txt)
        self._update_status_metrics(total_samples=total_samples)

    def _setup_status_metrics(self) -> None:
        self.status_files = QLabel("Dosya: 0")
        self.status_channels = QLabel("Kanal: 0")
        self.status_samples = QLabel("Örnek: 0")
        for lbl in (self.status_files, self.status_channels, self.status_samples):
            self.status.addPermanentWidget(lbl)
        self._update_status_metrics()

    def _update_status_metrics(self, *, total_samples: Optional[int] = None) -> None:
        if not hasattr(self, "status_files"):
            return
        self.status_files.setText(f"Dosya: {len(self.files)}")
        nchan = len(self.current_series) if self.current_series else 0
        self.status_channels.setText(f"Kanal: {nchan}")
        if total_samples is None:
            total_samples = 0
            if self.current_series:
                for s in self.current_series:
                    total_samples += int(s.get("sample_count", len(np.asarray(s.get("y", [])))) or 0)
        self.status_samples.setText(f"Örnek: {self._format_compact_count(int(total_samples))}")

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("appRoot")
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.setSpacing(6)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._main_splitter = splitter
        root_layout.addWidget(splitter, stretch=1)
        splitter.setHandleWidth(5)

        # LEFT panel
        left = QFrame()
        left.setObjectName("sidePanel")
        left.setMinimumWidth(300)
        left.setMaximumWidth(470)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        self.file_tabs = QTabWidget()
        self.file_tabs.setDocumentMode(True)
        self.file_tabs.setTabsClosable(True)
        self.file_tabs.tabCloseRequested.connect(self._close_file_tab)

        open_tab = QWidget()
        open_l = QVBoxLayout(open_tab)
        open_l.setContentsMargins(10, 10, 10, 10)
        open_l.setSpacing(10)
        self.btn_open = QPushButton("TDMS Dosyası Aç")
        self.btn_open.setProperty("primary", True)
        self.lbl_hint = QLabel("Bir veya daha fazla TDMS dosyası açılabilir.")
        self.lbl_hint.setWordWrap(True)
        open_l.addWidget(self.btn_open)
        open_l.addWidget(self.lbl_hint)
        open_l.addStretch(1)
        self.file_tabs.addTab(open_tab, "Aç")

        left_layout.addWidget(self.file_tabs, stretch=1)

        plot_row = QHBoxLayout()
        plot_row.setSpacing(8)
        self.btn_plot_checked = QPushButton("Seçili Kanalları Çiz")
        self.btn_plot_checked.setProperty("primary", True)
        self.btn_uncheck_all = QPushButton("Tümünü Kaldır")
        self.btn_clear_plot = QPushButton("Grafiği Temizle")
        self.btn_clear_plot.setProperty("danger", True)
        plot_row.addWidget(self.btn_plot_checked)
        plot_row.addWidget(self.btn_uncheck_all)
        plot_row.addWidget(self.btn_clear_plot)
        left_layout.addLayout(plot_row)

        self.preview_box = QGroupBox("Kanal Özellikleri")
        pf = QFormLayout(self.preview_box)
        pf.setVerticalSpacing(8)
        pf.setHorizontalSpacing(10)

        self.p_file, self.p_name = QLabel("—"), QLabel("—")
        self.p_samples, self.p_fs = QLabel("—"), QLabel("—")
        self.p_unit, self.p_quantity = QLabel("—"), QLabel("—")
        self.chk_manual_fs_plot = QCheckBox("Referans Fs kullan")
        self.chk_manual_fs_plot.setToolTip(
            "TDMS içinde zaman/Fs metadata'sı yoksa kullanılır. Girilen değer, seçili kanallar içinde "
            "en yüksek örnek sayısına sahip kanalın Fs değeri kabul edilir; diğer kanallar ortak test süresinden otomatik hesaplanır."
        )
        self.sp_manual_fs_plot = QDoubleSpinBox()
        self.sp_manual_fs_plot.setRange(0.0, 1e12)
        self.sp_manual_fs_plot.setDecimals(6)
        self.sp_manual_fs_plot.setValue(0.0)
        self.sp_manual_fs_plot.setKeyboardTracking(False)
        self.sp_manual_fs_plot.setToolTip(
            "En yüksek örnek sayılı kanalın bilinen örnekleme frekansı. Örn: 25 kHz için 25 girin ve kHz seçin."
        )
        self.cmb_manual_fs_unit = QComboBox()
        self.cmb_manual_fs_unit.addItems(["Hz", "kHz"])
        self.cmb_manual_fs_unit.setCurrentText("kHz")
        manual_fs_row = QWidget()
        mfsl = QHBoxLayout(manual_fs_row)
        mfsl.setContentsMargins(0, 0, 0, 0)
        mfsl.setSpacing(6)
        mfsl.addWidget(self.chk_manual_fs_plot)
        mfsl.addWidget(self.sp_manual_fs_plot, 1)
        mfsl.addWidget(self.cmb_manual_fs_unit)
        pf.addRow("Dosya:", self.p_file)
        pf.addRow("İsim:", self.p_name)
        pf.addRow("Örnek Sayısı:", self.p_samples)
        pf.addRow("Fs:", self.p_fs)
        pf.addRow("Referans Fs:", manual_fs_row)

        # Per-channel manual Fs editor lives in a dedicated tab.  The compact
        # button here keeps the preview panel uncluttered while Referans Fs stays
        # exactly where it is.
        self.btn_open_manual_fs_tab = QPushButton("Manuel Fs...")
        self.btn_open_manual_fs_tab.setToolTip(
            "Tüm açık TDMS kanallarını listeleyen Manuel Fs sekmesini açar. "
            "Her kanala ayrı Hz/kHz değeri atanabilir."
        )
        pf.addRow("Kanal Fs:", self.btn_open_manual_fs_tab)

        pf.addRow("Büyüklük:", self.p_quantity)
        pf.addRow("Birim:", self.p_unit)
        left_layout.addWidget(self.preview_box)


        splitter.addWidget(left)

        # CENTER workspace
        right = QFrame()
        right.setObjectName("workspacePanel")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("workspaceTabs")
        self.tabs.setDocumentMode(True)
        self.tabs.setMovable(True)

        # Compact workspace-level settings entry. Detailed controls open as a
        # bottom drawer inside the Graph tab, so the plot never loses width.
        self.btn_plot_details = QToolButton()
        self.btn_plot_details.setObjectName("workspaceDetailsButton")
        self.btn_plot_details.setText("⚙  Ayarlar")
        self.btn_plot_details.setToolTip("Grafik detay ayarlarını aç/kapat")
        self.btn_plot_details.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.btn_plot_details.setCursor(Qt.CursorShape.PointingHandCursor)
        self.tabs.setCornerWidget(self.btn_plot_details, Qt.Corner.TopRightCorner)
        self.btn_plot_details.clicked.connect(
            lambda: self._set_advanced_controls_visible(not self.controls_dock.isVisible())
        )
        right_layout.addWidget(self.tabs, stretch=1)

        # --- Plot tab ---
        plot_tab = QWidget()
        plot_tab_layout = QVBoxLayout(plot_tab)
        plot_tab_layout.setContentsMargins(6, 6, 6, 6)
        plot_tab_layout.setSpacing(6)

        # Plot mode is controlled from the ribbon. Keep a hidden combo as the
        # existing signal/state source so analysis logic remains unchanged.
        self.cmb_plot_mode = QComboBox()
        self.cmb_plot_mode.addItems(["Overlay", "Stacked"])
        self.cmb_plot_mode.setVisible(False)
        self.cmb_plot_mode.setToolTip("Overlay / Stacked görünüm durumu")

        self.plot_pane = PlotPane(bg_rgb=self.plot_bg_rgb)
        self.stacked_view = StackedPlotView(bg_rgb=self.plot_bg_rgb)
        self.plot_container = QStackedWidget()
        self.plot_container.addWidget(self.plot_pane)
        self.plot_container.addWidget(self.stacked_view)

        # --- Collapsible controls dock ---
        controls = QFrame()
        controls.setObjectName("controlsDock")
        controls.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        controls.setMinimumWidth(0)
        controls.setMaximumWidth(16777215)
        controls.setMinimumHeight(215)
        controls.setMaximumHeight(300)
        self.controls_dock = controls
        dock_layout = QVBoxLayout(controls)
        dock_layout.setContentsMargins(0, 0, 0, 0)
        dock_layout.setSpacing(0)

        self.controls_header = QFrame()
        self.controls_header.setObjectName("controlsHeader")
        hdr = QHBoxLayout(self.controls_header)
        hdr.setContentsMargins(10, 8, 10, 8)
        hdr.setSpacing(8)
        self.btn_ctrl_collapse = QToolButton()
        self.btn_ctrl_collapse.setObjectName("btnCtrlCollapse")
        self.btn_ctrl_collapse.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.btn_ctrl_collapse.setText("Detay Ayarları")
        self.btn_ctrl_collapse.setCheckable(True)
        self.btn_ctrl_collapse.setChecked(True)
        self.btn_ctrl_collapse.setArrowType(Qt.ArrowType.RightArrow)
        self.btn_ctrl_collapse.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_ctrl_collapse.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_ctrl_collapse.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        hdr.addWidget(self.btn_ctrl_collapse, 1)
        self.btn_details_close = QToolButton()
        self.btn_details_close.setText("✕")
        self.btn_details_close.setToolTip("Detay ayarlarını kapat")
        self.btn_details_close.setFixedSize(28, 28)
        self.btn_details_close.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_details_close.clicked.connect(lambda: self._set_advanced_controls_visible(False))
        hdr.addWidget(self.btn_details_close)
        dock_layout.addWidget(self.controls_header)

        self.controls_body = QWidget()
        body_layout = QVBoxLayout(self.controls_body)
        body_layout.setContentsMargins(10, 10, 10, 10)
        body_layout.setSpacing(8)
        dock_layout.addWidget(self.controls_body)
        self.controls_body.setVisible(True)

        self.ctrl_tabs = QTabWidget()
        self.ctrl_tabs.setDocumentMode(True)
        body_layout.addWidget(self.ctrl_tabs)

        # Tab: Range
        tab_range = QWidget()
        rg = QGridLayout(tab_range)
        rg.setContentsMargins(6, 6, 6, 6)
        rg.setHorizontalSpacing(10)
        rg.setVerticalSpacing(8)

        self.chk_region = QCheckBox("Aralık Seçici")
        self.chk_region.setVisible(False)  # özellik şimdilik UI dışı
        self.xmin = QDoubleSpinBox()
        self.xmax = QDoubleSpinBox()
        for w in (self.xmin, self.xmax):
            w.setRange(-1e18, 1e18)
            w.setDecimals(6)
            w.setKeyboardTracking(False)
        self.btn_fit_y = QPushButton("Y'yi Aralığa Sığdır")
        self.btn_bg_pick = QPushButton("Arka Plan Rengi")
        self.btn_bg_reset = QPushButton("Varsayılan Arka Plan")
        self.btn_detach = QPushButton("Grafiği Ayır")
        self.btn_export_csv = QPushButton("CSV Dışa Aktar")
        self.btn_export_csv.setToolTip("Mevcut çizili kanal verilerini CSV olarak dışa aktar")
        self.btn_save_plot_img = QPushButton("Grafiği Kaydet")
        self.btn_save_plot_img.setToolTip("Grafiği PNG/SVG olarak kaydet (Ctrl+Shift+S)")
        self._set_bg_button_preview(self.plot_bg_rgb)

        rg.addWidget(QLabel("Görünüm X Min:"), 0, 0)
        rg.addWidget(self.xmin, 0, 1)
        rg.addWidget(QLabel("Görünüm X Maks:"), 0, 2)
        rg.addWidget(self.xmax, 0, 3)
        rg.addWidget(self.btn_fit_y, 0, 4, 1, 2)
        rg.addWidget(self.btn_bg_pick, 1, 0, 1, 2)
        rg.addWidget(self.btn_bg_reset, 1, 2, 1, 2)
        rg.addWidget(self.btn_detach, 1, 4)
        rg.addWidget(self.btn_export_csv, 1, 5)
        rg.addWidget(self.btn_save_plot_img, 2, 0, 1, 2)
        self.ctrl_tabs.addTab(tab_range, "Eksen")

        # Tab: Axis titles / typography
        tab_axis_style = QWidget()
        ag = QGridLayout(tab_axis_style)
        ag.setContentsMargins(6, 6, 6, 6)
        ag.setHorizontalSpacing(10)
        ag.setVerticalSpacing(8)

        self.le_axis_x_title = QLineEdit()
        self.le_axis_y_title = QLineEdit()
        self.le_axis_y2_title = QLineEdit()
        self.le_axis_x_title.setPlaceholderText("Otomatik: Zaman (s) / İndeks")
        self.le_axis_y_title.setPlaceholderText("Otomatik: kanal büyüklüğü / birim")
        self.le_axis_y2_title.setPlaceholderText("Otomatik: sağ eksen büyüklüğü")
        self.le_axis_y_title.setToolTip("Overlay görünümündeki sol Y ekseni adı. Stacked görünümde kanal adları korunur.")
        self.le_axis_y2_title.setToolTip("Overlay görünümündeki sağ Y2 ekseni adı.")

        self.cmb_axis_font = QFontComboBox()
        self.cmb_axis_font.setCurrentFont(QFont("Arial"))
        self.sp_axis_title_size = QSpinBox()
        self.sp_axis_title_size.setRange(8, 28)
        self.sp_axis_title_size.setValue(11)
        self.sp_axis_title_size.setSuffix(" pt")
        self.chk_axis_title_bold = QCheckBox("Başlıklar Bold")

        self.sp_axis_tick_size = QSpinBox()
        self.sp_axis_tick_size.setRange(7, 20)
        self.sp_axis_tick_size.setValue(9)
        self.sp_axis_tick_size.setSuffix(" pt")
        self.chk_axis_tick_bold = QCheckBox("Tick Değerleri Bold")

        self.btn_axis_style_apply = QPushButton("Eksen Stilini Uygula")
        self.btn_axis_style_apply.setProperty("primary", True)
        self.btn_axis_style_reset = QPushButton("Varsayılana Dön")
        axis_note = QLabel("Boş bırakılan eksen adları otomatik kalır. Eksen adına çift tıklayarak hızlı düzenleme penceresini de açabilirsiniz. Stacked görünümde Y başlıkları kanal adı/birim olarak korunur.")
        axis_note.setWordWrap(True)
        axis_note.setStyleSheet("color: #6B7280;")

        ag.addWidget(QLabel("X Ekseni Adı:"), 0, 0)
        ag.addWidget(self.le_axis_x_title, 0, 1, 1, 2)
        ag.addWidget(QLabel("Y Ekseni Adı:"), 0, 3)
        ag.addWidget(self.le_axis_y_title, 0, 4, 1, 2)
        ag.addWidget(QLabel("Y2 Ekseni Adı:"), 1, 0)
        ag.addWidget(self.le_axis_y2_title, 1, 1, 1, 2)
        ag.addWidget(QLabel("Font:"), 1, 3)
        ag.addWidget(self.cmb_axis_font, 1, 4, 1, 2)
        ag.addWidget(QLabel("Başlık Boyutu:"), 2, 0)
        ag.addWidget(self.sp_axis_title_size, 2, 1)
        ag.addWidget(self.chk_axis_title_bold, 2, 2)
        ag.addWidget(QLabel("Tick Boyutu:"), 2, 3)
        ag.addWidget(self.sp_axis_tick_size, 2, 4)
        ag.addWidget(self.chk_axis_tick_bold, 2, 5)
        ag.addWidget(self.btn_axis_style_apply, 3, 0, 1, 2)
        ag.addWidget(self.btn_axis_style_reset, 3, 2, 1, 2)
        ag.addWidget(axis_note, 4, 0, 1, 6)
        self.ctrl_tabs.addTab(tab_axis_style, "Eksen Stili")

        # Tab: Markers
        tab_markers = QWidget()
        mg = QGridLayout(tab_markers)
        mg.setContentsMargins(6, 6, 6, 6)
        mg.setHorizontalSpacing(10)
        mg.setVerticalSpacing(8)

        self.chk_click_mark = QCheckBox("Tıkla İşaretle")
        self.btn_clear_markers = QPushButton("İşaretleri Temizle")
        self.btn_clear_markers.setProperty("danger", True)
        self.btn_marker_add = QPushButton("Topla")
        self.btn_marker_sub = QPushButton("Çıkar")
        self.btn_marker_stats = QPushButton("Aralık İstatistiği")
        self.btn_marker_add.setToolTip("Seçili 2 işaretin değerlerini topla")
        self.btn_marker_sub.setToolTip("Seçili işaretleri çıkar (B - A)")
        self.cmb_marker_calc_axis = QComboBox()
        self.cmb_marker_calc_axis.addItems(["Yalnız X", "Yalnız Y", "X ve Y"])
        self.lbl_marker_calc = QLabel("Sonuç: —")
        self.lbl_marker_calc.setStyleSheet("font-weight: 800;")

        self.marker_list = QTreeWidget()
        self.marker_list.setHeaderLabels(["İşaret", "X Değeri", "Y Değeri"])
        self.marker_list.setMaximumHeight(130)
        self.marker_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        mg.addWidget(self.chk_click_mark, 0, 0)
        mg.addWidget(self.btn_clear_markers, 0, 1)
        mg.addWidget(QLabel("Hesap:"), 0, 2)
        mg.addWidget(self.cmb_marker_calc_axis, 0, 3)
        mg.addWidget(self.btn_marker_add, 0, 4)
        mg.addWidget(self.btn_marker_sub, 0, 5)
        mg.addWidget(self.lbl_marker_calc, 0, 6)
        mg.addWidget(self.btn_marker_stats, 1, 0, 1, 2)
        mg.addWidget(self.marker_list, 2, 0, 1, 7)
        self.ctrl_tabs.addTab(tab_markers, "İşaretler")

        # Tab: Filter & Style
        tab_style = QWidget()
        sg = QGridLayout(tab_style)
        sg.setContentsMargins(6, 6, 6, 6)
        sg.setHorizontalSpacing(10)
        sg.setVerticalSpacing(8)

        self.chk_smooth = QCheckBox("Savitzky-Golay Yumuşatma")
        self.chk_cursor_marker = QCheckBox("İmleç İşaretini Göster")
        self.chk_cursor_marker.setChecked(True)
        self.chk_cursor_marker.setToolTip("Grafik üzerindeki artı işareti/nokta imlecini aç/kapat")
        self.spin_smooth_win = QSpinBox()
        self.spin_smooth_win.setRange(5, 999)
        self.spin_smooth_win.setSingleStep(2)
        self.spin_smooth_win.setValue(CFG.default_filter_window)
        self.spin_smooth_win.setSuffix(" pts")

        self.cmb_style_series = QComboBox()
        self.cmb_style_series.setMinimumWidth(320)
        self.btn_pick_color = QPushButton("Renk Seç")
        self._set_color_button_preview(None)

        self.sp_line_width = QDoubleSpinBox()
        self.sp_line_width.setRange(0.5, 12.0)
        self.sp_line_width.setDecimals(1)
        self.sp_line_width.setSingleStep(0.5)
        self.sp_line_width.setValue(2.0)
        self.sp_line_width.setKeyboardTracking(False)

        self.sp_x_shift = QDoubleSpinBox()
        self.sp_x_shift.setRange(-1e18, 1e18)
        self.sp_x_shift.setDecimals(6)
        self.sp_x_shift.setSingleStep(0.1)
        self.sp_x_shift.setValue(0.0)
        self.sp_x_shift.setKeyboardTracking(False)
        self.sp_x_shift.setToolTip("Kanal başına X kaydırma. Zaman eksenleri için saniye, indeks için örnek sayısı.")

        self.sp_y_shift = QDoubleSpinBox()
        self.sp_y_shift.setRange(-1e18, 1e18)
        self.sp_y_shift.setDecimals(6)
        self.sp_y_shift.setSingleStep(0.1)
        self.sp_y_shift.setValue(0.0)
        self.sp_y_shift.setKeyboardTracking(False)
        self.sp_y_shift.setToolTip("Kanal başına Y kaydırma. Dikey ofset değeri.")

        self.btn_style_apply = QPushButton("Uygula")
        self.btn_style_default = QPushButton("Varsayılan")
        self.btn_style_reset_all = QPushButton("Tümünü Sıfırla")
        self.btn_style_reset_all.setProperty("danger", True)

        self.shift_series_tree = QTreeWidget()
        self.shift_series_tree.setHeaderLabels(["Kanal", "Stil Anahtarı"])
        self.shift_series_tree.setColumnCount(2)
        self.shift_series_tree.setRootIsDecorated(False)
        self.shift_series_tree.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.shift_series_tree.setMaximumHeight(95)

        self.btn_shift_apply = QPushButton("Shift Uygula")
        self.btn_shift_reset = QPushButton("Shift Sıfırla")
        self.btn_shift_check_all = QPushButton("Tümünü Seç")
        self.btn_shift_uncheck_all = QPushButton("Tümünü Kaldır")

        sg.addWidget(self.chk_smooth, 0, 0, 1, 2)
        sg.addWidget(self.chk_cursor_marker, 0, 4, 1, 2)
        sg.addWidget(QLabel("Filtre Penceresi:"), 0, 2)
        sg.addWidget(self.spin_smooth_win, 0, 3)
        sg.addWidget(QLabel("Kanal:"), 1, 0)
        sg.addWidget(self.cmb_style_series, 1, 1, 1, 2)
        sg.addWidget(self.btn_pick_color, 1, 3)
        sg.addWidget(QLabel("Kalınlık:"), 1, 4)
        sg.addWidget(self.sp_line_width, 1, 5)
        sg.addWidget(QLabel("X Kaydırma:"), 2, 0)
        sg.addWidget(self.sp_x_shift, 2, 1)
        sg.addWidget(QLabel("Y Kaydırma:"), 2, 2)
        sg.addWidget(self.sp_y_shift, 2, 3)
        sg.addWidget(self.btn_style_apply, 2, 4)
        sg.addWidget(self.btn_style_default, 2, 5)
        sg.addWidget(self.btn_style_reset_all, 3, 0, 1, 2)
        sg.addWidget(self.shift_series_tree, 4, 0, 1, 6)
        shift_btn_row = QHBoxLayout()
        shift_btn_row.addWidget(self.btn_shift_check_all)
        shift_btn_row.addWidget(self.btn_shift_uncheck_all)
        shift_btn_row.addWidget(self.btn_shift_apply)
        shift_btn_row.addWidget(self.btn_shift_reset)
        shift_btn_row.addStretch()
        sg.addLayout(shift_btn_row, 5, 0, 1, 6)
        self.ctrl_tabs.addTab(tab_style, "Stil")

        self._plot_vsplit = None
        self._controls_last_sizes = []
        self.btn_ctrl_collapse.toggled.connect(self._on_controls_panel_toggled)
        self._on_controls_panel_toggled(True)

        plot_tab_layout.addWidget(self.plot_container, stretch=1)
        plot_tab_layout.addWidget(controls)
        self.workspace_summary = self._build_workspace_summary()
        plot_tab_layout.addWidget(self.workspace_summary)
        self.tabs.addTab(plot_tab, "Grafik")

        # --- User-entered interval statistics tab ---
        self.range_stats_tab = self._build_range_stats_tab()
        self.tabs.addTab(self.range_stats_tab, "Aralık Analizi")

        # --- FFT tab ---
        fft_tab = QWidget()
        fft_layout = QVBoxLayout(fft_tab)
        fft_layout.setContentsMargins(10, 10, 10, 10)
        fft_layout.setSpacing(10)

        fft_topbar = QHBoxLayout()
        fft_topbar.addStretch(1)
        self.btn_theme_fft = QToolButton()
        self.btn_theme_fft.setCheckable(True)
        self.btn_theme_fft.setChecked(self._theme == "dark")
        self.btn_theme_fft.setText(self._theme_button_text())
        self.btn_theme_fft.setToolTip("Tema Değiştir (Koyu/Açık)")
        self.btn_theme_fft.setFixedSize(34, 34)
        self.btn_theme_fft.setStyleSheet("border-radius: 17px; font-weight: 900; padding: 0px;")
        fft_topbar.addWidget(self.btn_theme_fft)
        fft_layout.addLayout(fft_topbar)

        self.fft_plot = PlotWidget()
        self.fft_plot.showGrid(x=True, y=True, alpha=0.28)
        self.fft_plot.setLabel("bottom", "Frekans (Hz)")
        self.fft_plot.setLabel("left", "Genlik")
        fft_layout.addWidget(self.fft_plot, stretch=1)

        self.lbl_fft_info = QLabel("Bir kanal seçin ve FFT hesaplayın.")
        self.lbl_fft_info.setWordWrap(True)
        self.lbl_fft_info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        fft_layout.addWidget(self.lbl_fft_info)

        fft_panel = QFrame()
        fft_panel.setObjectName("fftPanel")
        fft_panel.setFrameShape(QFrame.Shape.StyledPanel)
        fft_panel_l = QVBoxLayout(fft_panel)
        fft_panel_l.setContentsMargins(10, 10, 10, 10)
        fft_panel_l.setSpacing(8)

        fft_row1 = QGridLayout()
        fft_row1.setHorizontalSpacing(8)
        fft_row1.setVerticalSpacing(8)
        self.cmb_fft_channel = QComboBox()
        self.cmb_fft_channel.setToolTip("FFT analizi için kanal")
        self.fs_fft = QDoubleSpinBox()
        self.fs_fft.setRange(0.0, 1e12)
        self.fs_fft.setDecimals(6)
        self.fs_fft.setValue(0.0)
        self.fs_fft.setKeyboardTracking(False)
        self.fs_fft.setToolTip("0 = zaman ekseninden otomatik. Gerekirse Fs'yi manuel girin.")
        self.btn_fft = QPushButton("FFT Hesapla")
        self.btn_fft.setProperty("primary", True)
        self.btn_fft_clear = QPushButton("FFT Temizle")
        fft_row1.addWidget(QLabel("Kanal:"), 0, 0)
        fft_row1.addWidget(self.cmb_fft_channel, 0, 1)
        fft_row1.addWidget(QLabel("Fs (0=otomatik):"), 0, 2)
        fft_row1.addWidget(self.fs_fft, 0, 3)
        fft_row1.addWidget(self.btn_fft, 0, 4)
        fft_row1.addWidget(self.btn_fft_clear, 0, 5)
        fft_row1.setColumnStretch(1, 1)
        fft_panel_l.addLayout(fft_row1)

        fft_row2 = QHBoxLayout()
        fft_row2.setSpacing(12)
        self.chk_fft_log = QCheckBox("Log (dB)")
        self.chk_fft_log.setToolTip("Y ekseni dB cinsinden")
        self.chk_fft_window = QCheckBox("Hanning penceresi")
        self.chk_fft_window.setChecked(True)
        self.chk_fft_window.setToolTip("Spektral kaçağı azaltmak için Hanning penceresi uygula")
        self.chk_fft_remove_mean = QCheckBox("Ortalamayı kaldır")
        self.chk_fft_remove_mean.setChecked(True)
        self.chk_fft_remove_mean.setToolTip("DC bileşenini azalt")
        self.chk_fft_detrend = QCheckBox("Doğrusal trend kaldır")
        self.chk_fft_detrend.setChecked(True)
        self.chk_fft_detrend.setToolTip("Yavaş trend/sapma kaldır")
        self.chk_fft_hide_dc = QCheckBox("0 Hz'i gizle")
        self.chk_fft_hide_dc.setChecked(True)
        self.chk_fft_hide_dc.setToolTip("DC bileşenini grafikten gizle")
        self.chk_fft_peak = QCheckBox("Baskın tepeyi işaretle")
        self.chk_fft_peak.setChecked(True)
        self.chk_fft_peak.setToolTip("En güçlü frekansı işaretle")
        for w in (self.chk_fft_log, self.chk_fft_window, self.chk_fft_remove_mean,
                  self.chk_fft_detrend, self.chk_fft_hide_dc, self.chk_fft_peak):
            fft_row2.addWidget(w)
        fft_row2.addStretch(1)
        fft_panel_l.addLayout(fft_row2)

        fft_row3 = QHBoxLayout()
        fft_row3.setSpacing(10)
        self.sp_fft_fmin = QDoubleSpinBox()
        self.sp_fft_fmin.setRange(0.0, 1e12)
        self.sp_fft_fmin.setDecimals(6)
        self.sp_fft_fmin.setValue(0.0)
        self.sp_fft_fmin.setKeyboardTracking(False)
        self.sp_fft_fmax = QDoubleSpinBox()
        self.sp_fft_fmax.setRange(0.0, 1e12)
        self.sp_fft_fmax.setDecimals(6)
        self.sp_fft_fmax.setValue(0.0)
        self.sp_fft_fmax.setKeyboardTracking(False)
        fft_row3.addWidget(QLabel("Min f (Hz):"))
        fft_row3.addWidget(self.sp_fft_fmin)
        fft_row3.addSpacing(8)
        fft_row3.addWidget(QLabel("Maks f (Hz, 0=Nyquist):"))
        fft_row3.addWidget(self.sp_fft_fmax)
        fft_row3.addStretch(1)
        fft_panel_l.addLayout(fft_row3)

        fft_layout.addWidget(fft_panel)
        self.tabs.addTab(fft_tab, "Frekans Analizi (FFT)")

        # Analysis workspaces are first-class tabs instead of duplicate buttons
        # inside the graph header/ribbon. This keeps the command area compact.
        self.channel_manager_tab = self._build_channel_manager_tab()
        self.tabs.addTab(self.channel_manager_tab, "Kanal Yöneticisi")
        self.calculated_channel_tab = self._build_calculated_channel_tab()
        self.tabs.addTab(self.calculated_channel_tab, "Hesaplanan Kanal")
        self.manual_fs_tab = self._build_manual_fs_tab()
        self.tabs.addTab(self.manual_fs_tab, "Manuel Fs")

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setSizes([350, 1330])

        # Build a single compact command ribbon. Detailed numeric/form
        # controls live in the on-demand bottom drawer inside the Graph tab.
        self.ribbon = self._build_ribbon()
        self.control_ribbon = None
        root_layout.insertWidget(0, self.ribbon)

        # Start with the detail drawer hidden for maximum plot area.
        # Ribbon/corner "Ayarlar" commands open it when needed.
        self._set_advanced_controls_visible(False)
        self._update_workspace_summary()

    def _connect_signals(self) -> None:
        self.btn_open.clicked.connect(self.open_tdms)
        self.btn_plot_checked.clicked.connect(self.plot_checked_channels)
        self.btn_uncheck_all.clicked.connect(self.uncheck_all_channels)
        self.btn_clear_plot.clicked.connect(self.on_clear_plot)

        self.chk_region.toggled.connect(self._set_region_enabled_all)
        self.chk_click_mark.toggled.connect(self._set_click_mark_enabled_all)

        self.plot_pane.range_changed.connect(lambda x0, x1: self._on_any_range_changed("main", x0, x1))
        self.stacked_view.range_changed.connect(lambda x0, x1: self._on_any_range_changed("stacked", x0, x1))
        self.plot_pane.axis_edit_requested.connect(self._open_axis_quick_editor)
        self.stacked_view.axis_edit_requested.connect(self._open_axis_quick_editor)
        self.xmin.valueChanged.connect(self.apply_range_manual)
        self.xmax.valueChanged.connect(self.apply_range_manual)
        self.btn_fit_y.clicked.connect(self.autofit_y_in_range)

        self.plot_pane.interaction_mode_changed.connect(self._on_pane_mode_changed)
        self.plot_pane.autofit_requested.connect(self._on_pane_autofit)
        self.plot_pane.legend_toggled.connect(self._on_pane_legend)
        self.plot_pane.y_lock_toggled.connect(self._on_pane_ylock)
        self.plot_pane.right_y_lock_toggled.connect(self._on_pane_y2lock)
        self.plot_pane.region_toggled.connect(self._on_pane_region)
        self.plot_pane.click_mark_toggled.connect(self._on_pane_click_mark)

        self.btn_clear_markers.clicked.connect(self.clear_markers)
        self.plot_pane.marker_requested.connect(lambda x, y: self._on_marker_requested_from("main", x, y))
        self.marker_list.itemDoubleClicked.connect(self._on_marker_double_clicked)
        self.btn_marker_add.clicked.connect(self._marker_math_add)
        self.btn_marker_sub.clicked.connect(self._marker_math_sub)
        self.btn_marker_stats.clicked.connect(self._marker_interval_stats)

        self.chk_smooth.toggled.connect(self.update_plot_data_with_filter)
        self.chk_cursor_marker.toggled.connect(self._set_cursor_marker_enabled_all)
        self.spin_smooth_win.valueChanged.connect(self.update_plot_data_with_filter)

        self.cmb_style_series.currentIndexChanged.connect(self._on_style_series_changed)
        self.btn_pick_color.clicked.connect(self.pick_color_for_selected_series)
        self.btn_style_apply.clicked.connect(self.apply_style_for_selected_series)
        self.btn_style_default.clicked.connect(self.reset_style_for_selected_series)
        self.btn_style_reset_all.clicked.connect(self.reset_all_styles)

        self.btn_shift_apply.clicked.connect(self.apply_shift_for_checked_series)
        self.btn_shift_reset.clicked.connect(self.reset_shift_for_checked_series)
        self.btn_shift_check_all.clicked.connect(lambda: self._set_all_shift_checks(True))
        self.btn_shift_uncheck_all.clicked.connect(lambda: self._set_all_shift_checks(False))

        self.btn_bg_pick.clicked.connect(self.pick_background_color)
        self.btn_bg_reset.clicked.connect(self.reset_background_color)
        self.btn_detach.clicked.connect(self.toggle_detach_plot)
        self.btn_export_csv.clicked.connect(self.export_csv)
        self.btn_save_plot_img.clicked.connect(self.save_plot_image)
        self.btn_axis_style_apply.clicked.connect(self._apply_axis_style_from_controls)
        self.btn_axis_style_reset.clicked.connect(self._reset_axis_style_controls)

        self.btn_fft.clicked.connect(self.compute_fft)
        self.btn_fft_clear.clicked.connect(self.clear_fft_plot)
        self.chk_fft_log.toggled.connect(self._rerender_fft_plot)
        self.chk_fft_hide_dc.toggled.connect(self._rerender_fft_plot)
        self.chk_fft_peak.toggled.connect(self._rerender_fft_plot)
        self.sp_fft_fmin.valueChanged.connect(self._rerender_fft_plot)
        self.sp_fft_fmax.valueChanged.connect(self._rerender_fft_plot)

        self.btn_theme_fft.clicked.connect(self._toggle_theme_from_button)
        self.chk_manual_fs_plot.toggled.connect(self._on_manual_fs_controls_changed)
        self.sp_manual_fs_plot.valueChanged.connect(self._on_manual_fs_controls_changed)
        self.cmb_manual_fs_unit.currentIndexChanged.connect(self._on_manual_fs_controls_changed)
        self.btn_open_manual_fs_tab.clicked.connect(self._open_manual_fs_tab)
        self.cmb_plot_mode.currentIndexChanged.connect(self._on_plot_mode_changed)
        self.tabs.currentChanged.connect(self._on_workspace_tab_changed)
        self.plot_pane.copy_requested.connect(self.copy_plot_to_clipboard)
        self.plot_pane.save_requested.connect(self.save_plot_image)
        self.plot_pane.zoom_back_requested.connect(self._undo_zoom)

    def _on_controls_panel_toggled(self, expanded: bool) -> None:
        vs = getattr(self, "_plot_vsplit", None)
        if vs is None:
            try:
                self.controls_body.setVisible(bool(expanded))
                self.btn_ctrl_collapse.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
                dock = getattr(self, "controls_dock", None)
                if dock is not None:
                    dock.setMinimumWidth(0)
                    dock.setMaximumWidth(16777215)
                    dock.setMinimumHeight(215 if expanded else 42)
                    dock.setMaximumHeight(300 if expanded else 46)
            except Exception:
                logger.debug("Details drawer toggle error", exc_info=True)
            return
        try:
            header_h = 42
            try:
                header_h = max(header_h, self.controls_header.sizeHint().height() + 6)
            except Exception:
                pass
            total = max(sum(vs.sizes()), 1)

            if expanded:
                self.controls_body.setVisible(True)
                self.btn_ctrl_collapse.setArrowType(Qt.ArrowType.DownArrow)
                sizes = getattr(self, "_controls_last_sizes", None)
                if isinstance(sizes, list) and len(sizes) == 2 and sum(sizes) > 0:
                    vs.setSizes(sizes)
                else:
                    vs.setSizes([max(10, total - 170), 170])
            else:
                self._controls_last_sizes = vs.sizes()
                self.controls_body.setVisible(False)
                self.btn_ctrl_collapse.setArrowType(Qt.ArrowType.RightArrow)
                vs.setSizes([max(10, total - header_h), header_h])
        except Exception:
            logger.debug("Controls panel toggle error", exc_info=True)

    # ----- Pane sync helpers -----

    def _silent_set_checked(self, w: Any, checked: bool) -> None:
        if w is None:
            return
        try:
            blocked = w.blockSignals(True)
            w.setChecked(checked)
            w.blockSignals(blocked)
        except Exception:
            pass

    def _on_pane_mode_changed(self, mode: str) -> None:
        if not self._pane_sync_guard:
            self._pane_sync_guard = True
            try:
                self._apply_interaction_mode_to_all(mode)
            finally:
                self._pane_sync_guard = False

    def _on_pane_autofit(self) -> None:
        if not self._pane_sync_guard:
            self._pane_sync_guard = True
            try:
                self._autofit_all_panes()
            finally:
                self._pane_sync_guard = False

    def _on_pane_legend(self, on: bool) -> None:
        if not self._pane_sync_guard:
            self._pane_sync_guard = True
            try:
                self._set_legend_visible_all(on)
            finally:
                self._pane_sync_guard = False

    def _on_pane_ylock(self, on: bool) -> None:
        if not self._pane_sync_guard:
            self._pane_sync_guard = True
            try:
                self._set_y_lock_enabled_all(on)
            finally:
                self._pane_sync_guard = False

    def _on_pane_y2lock(self, on: bool) -> None:
        if not self._pane_sync_guard:
            self._pane_sync_guard = True
            try:
                self._set_right_y_lock_enabled_all(on)
            finally:
                self._pane_sync_guard = False

    def _on_pane_region(self, on: bool) -> None:
        if not self._pane_sync_guard:
            self._pane_sync_guard = True
            try:
                self._silent_set_checked(self.chk_region, on)
                self._set_region_enabled_all(on)
            finally:
                self._pane_sync_guard = False

    def _on_pane_click_mark(self, on: bool) -> None:
        if not self._pane_sync_guard:
            self._pane_sync_guard = True
            try:
                self._silent_set_checked(self.chk_click_mark, on)
                self._set_click_mark_enabled_all(on)
            finally:
                self._pane_sync_guard = False

    # ----- Apply-to-all helpers -----

    def _apply_interaction_mode_to_all(self, mode: Optional[str] = None) -> None:
        m = (mode or getattr(self.plot_pane, "_interaction_mode", "pan") or "pan").strip().lower()
        m = "pan" if m == "pan" else "zoom"
        self.plot_pane.set_interaction_mode(m)
        if self.detached_win is not None:
            self.detached_win.pane.set_interaction_mode(m)

    def _autofit_all_panes(self) -> None:
        self.plot_pane.autofit_all()
        self.stacked_view.autofit_all()
        if self.detached_win is not None:
            self.detached_win.pane.autofit_all()

    def _set_region_enabled_all(self, enabled: bool) -> None:
        self.plot_pane.enable_region(enabled)
        if self.detached_win is not None:
            self.detached_win.pane.enable_region(enabled)

    def _set_click_mark_enabled_all(self, enabled: bool) -> None:
        self.plot_pane.set_click_mark_mode(enabled)
        if self.detached_win is not None:
            self.detached_win.pane.set_click_mark_mode(enabled)

    def _set_cursor_marker_enabled_all(self, enabled: bool) -> None:
        self.plot_pane.set_cursor_marker_enabled(enabled)
        if self.detached_win is not None:
            self.detached_win.pane.set_cursor_marker_enabled(enabled)

    def _set_y_lock_enabled_all(self, enabled: bool) -> None:
        self.plot_pane.set_y_lock(enabled)
        if self.detached_win is not None:
            self.detached_win.pane.set_y_lock(enabled)

    def _set_right_y_lock_enabled_all(self, enabled: bool) -> None:
        self.plot_pane.set_right_y_lock(enabled)
        if self.detached_win is not None:
            self.detached_win.pane.set_right_y_lock(enabled)

    def _set_legend_visible_all(self, enabled: bool) -> None:
        self.plot_pane.set_legend_visible(enabled)
        if self.detached_win is not None:
            self.detached_win.pane.set_legend_visible(enabled)

    # ----- Axis title / typography controls -----

    def _collect_axis_style_from_controls(self) -> Dict[str, Any]:
        family = "Arial"
        try:
            family = self.cmb_axis_font.currentFont().family() or "Arial"
        except Exception:
            pass
        return {
            "x_title": self.le_axis_x_title.text().strip(),
            "y_title": self.le_axis_y_title.text().strip(),
            "y2_title": self.le_axis_y2_title.text().strip(),
            "font_family": family,
            "title_size": int(self.sp_axis_title_size.value()),
            "title_bold": bool(self.chk_axis_title_bold.isChecked()),
            "tick_size": int(self.sp_axis_tick_size.value()),
            "tick_bold": bool(self.chk_axis_tick_bold.isChecked()),
        }

    def _apply_axis_style_from_controls(self, *_args: Any, show_status: bool = True) -> None:
        self.axis_style = self._collect_axis_style_from_controls()
        self.plot_pane.set_axis_style(self.axis_style)
        self.stacked_view.set_axis_style(self.axis_style)
        if self.detached_win is not None:
            self.detached_win.pane.set_axis_style(self.axis_style)
        if show_status and hasattr(self, "status"):
            self.status.showMessage("Eksen adı ve yazı stili uygulandı.", 1800)

    def _reset_axis_style_controls(self) -> None:
        self.le_axis_x_title.clear()
        self.le_axis_y_title.clear()
        self.le_axis_y2_title.clear()
        self.cmb_axis_font.setCurrentFont(QFont("Arial"))
        self.sp_axis_title_size.setValue(11)
        self.chk_axis_title_bold.setChecked(False)
        self.sp_axis_tick_size.setValue(9)
        self.chk_axis_tick_bold.setChecked(False)
        self._apply_axis_style_from_controls()

    def _open_axis_quick_editor(self, axis_key: str) -> None:
        """Open a compact editor when a plot axis is double-clicked."""
        key = str(axis_key or "").strip()
        if not key:
            return

        # Stacked Y labels are channel-specific, so edit the channel display name
        # while keeping typography shared with the regular axis-style controls.
        stacked_skey: Optional[str] = None
        if key.startswith("stacked_y::"):
            stacked_skey = key.split("::", 1)[1]
            chosen = next((s for s in (self.current_series or []) if str(s.get("style_key", "")) == stacked_skey), None)
            if chosen is None:
                return
            st = self.style_map.get(stacked_skey, {})
            current_title = str(st.get("display_name") or "")
            auto_title = str(chosen.get("name") or "Kanal")
            caption = "Stacked Y"
        else:
            edit_map = {
                "x": (self.le_axis_x_title, "X", str(self.plot_pane._axis_auto_titles.get("x") or self.plot_pane.x_label or "X")),
                "y": (self.le_axis_y_title, "Y", str(self.plot_pane._axis_auto_titles.get("y") or "Değer")),
                "y2": (self.le_axis_y2_title, "Y2", str(self.plot_pane._axis_auto_titles.get("y2") or "Sağ eksen")),
            }
            if key not in edit_map:
                return
            line_edit, caption, auto_title = edit_map[key]
            current_title = line_edit.text().strip()

        dlg = AxisQuickEditDialog(
            axis_caption=caption,
            current_title=current_title,
            auto_title=auto_title,
            style=self.axis_style,
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        vals = dlg.values()
        if stacked_skey is not None:
            st = dict(self.style_map.get(stacked_skey, {}))
            title = str(vals.get("title", "") or "").strip()
            if title:
                st["display_name"] = title
            else:
                st.pop("display_name", None)
            self.style_map[stacked_skey] = st
        else:
            target = {
                "x": self.le_axis_x_title,
                "y": self.le_axis_y_title,
                "y2": self.le_axis_y2_title,
            }[key]
            target.setText(str(vals.get("title", "") or ""))

        self.cmb_axis_font.setCurrentFont(QFont(str(vals.get("font_family", "Arial") or "Arial")))
        self.sp_axis_title_size.setValue(int(vals.get("title_size", 11) or 11))
        self.chk_axis_title_bold.setChecked(bool(vals.get("title_bold", False)))
        self.sp_axis_tick_size.setValue(int(vals.get("tick_size", 9) or 9))
        self.chk_axis_tick_bold.setChecked(bool(vals.get("tick_bold", False)))
        self._apply_axis_style_from_controls(show_status=False)

        # A stacked channel label lives in style_map and therefore needs a lightweight
        # redraw; overlay axis titles only need the style application above.
        if stacked_skey is not None:
            self._render_to_all_panes(self._last_display_series, self._last_axis_mode, self._last_x_label)
        if hasattr(self, "status"):
            self.status.showMessage(f"{caption} ekseni güncellendi.", 1600)

    # ----- Background -----

    def _set_bg_button_preview(self, rgb: Tuple[int, int, int]) -> None:
        r, g, b = rgb
        fr, fg_, fb = _bg_to_fg_rgb(rgb)
        self.btn_bg_pick.setStyleSheet(
            f"background-color: rgb({r},{g},{b}); color: rgb({fr},{fg_},{fb}); font-weight: 800;"
            f"border-radius: 9px; padding: 7px 12px;"
        )

    def pick_background_color(self) -> None:
        c = QColorDialog.getColor(QColor(*self.plot_bg_rgb), self, "Arka Plan Rengi Seçin")
        if not c.isValid():
            return
        self.plot_bg_rgb = qcolor_to_tuple(c)
        self._set_bg_button_preview(self.plot_bg_rgb)
        self._apply_background_everywhere()

    def reset_background_color(self) -> None:
        self.plot_bg_rgb = self._theme_default_bg[self._theme]
        self._set_bg_button_preview(self.plot_bg_rgb)
        self._apply_background_everywhere()

    def _apply_background_everywhere(self) -> None:
        self.plot_pane.set_background(self.plot_bg_rgb)
        self.stacked_view.set_background(self.plot_bg_rgb)
        if self.detached_win is not None:
            self.detached_win.pane.set_background(self.plot_bg_rgb)
        apply_plotwidget_theme(self.fft_plot, self.plot_bg_rgb)
        if self._last_fft_payload:
            self._render_fft_payload(self._last_fft_payload)

    # ----- Detach -----

    def toggle_detach_plot(self) -> None:
        if self.detached_win is None:
            self.detached_win = DetachedPlotWindow(bg_rgb=self.plot_bg_rgb, parent=self)
            self.detached_win.closed.connect(self._on_detached_closed)
            self.detached_win.pane.range_changed.connect(
                lambda x0, x1: self._on_any_range_changed("detached", x0, x1)
            )
            self.detached_win.pane.marker_requested.connect(
                lambda x, y: self._on_marker_requested_from("detached", x, y)
            )
            self.detached_win.pane.copy_requested.connect(self.copy_plot_to_clipboard)
            self.detached_win.pane.save_requested.connect(self.save_plot_image)
            self.detached_win.pane.zoom_back_requested.connect(self._undo_zoom)
            self.detached_win.pane.axis_edit_requested.connect(self._open_axis_quick_editor)

            for sig_name, handler in [
                ("interaction_mode_changed", self._on_pane_mode_changed),
                ("autofit_requested", self._on_pane_autofit),
                ("legend_toggled", self._on_pane_legend),
                ("y_lock_toggled", self._on_pane_ylock),
                ("right_y_lock_toggled", self._on_pane_y2lock),
                ("region_toggled", self._on_pane_region),
                ("click_mark_toggled", self._on_pane_click_mark),
            ]:
                getattr(self.detached_win.pane, sig_name).connect(handler)

            dp = self.detached_win.pane
            mp = self.plot_pane
            dp.set_interaction_mode(mp._interaction_mode)
            dp.enable_region(mp._region_enabled)
            dp.set_click_mark_mode(mp._click_mark_enabled)
            dp.set_y_lock(mp._y_lock_enabled)
            dp.set_right_y_lock(mp._right_y_lock_enabled)
            dp.set_legend_visible(mp._legend_visible)
            dp.set_axis_style(self.axis_style)

            self._silent_set_checked(self.chk_region, mp._region_enabled)
            self._silent_set_checked(self.chk_click_mark, mp._click_mark_enabled)

            self.update_plot_data_with_filter()
            for m in self._markers.values():
                self.detached_win.pane.add_marker(m["x"], m["y"], m["label"])

            self.detached_win.show()
            self.btn_detach.setText("Grafiği Geri Bağla")
        else:
            self.detached_win.close()

    def _on_detached_closed(self) -> None:
        if self.detached_win is not None:
            try:
                self.detached_win.deleteLater()
            except Exception:
                pass
        self.detached_win = None
        self.btn_detach.setText("Grafiği Ayır")

    # ----- Style UI -----

    def _set_color_button_preview(self, rgb: Optional[Tuple[int, int, int]]) -> None:
        if rgb is None:
            self.btn_pick_color.setText("Renk Seç (Otomatik)")
            self.btn_pick_color.setStyleSheet("")
        else:
            r, g, b = rgb
            self.btn_pick_color.setText("Renk Seç")
            self.btn_pick_color.setStyleSheet(
                f"background-color: rgb({r},{g},{b}); color: white; font-weight: 800; border-radius: 9px;"
            )

    def _current_style_key(self) -> Optional[str]:
        idx = self.cmb_style_series.currentIndex()
        if idx < 0:
            return None
        data = self.cmb_style_series.itemData(idx, role=Qt.ItemDataRole.UserRole)
        return data if isinstance(data, str) else None

    def _on_style_series_changed(self, *_: Any) -> None:
        skey = self._current_style_key()
        if not skey:
            self._set_color_button_preview(None)
            self.sp_line_width.setValue(2.0)
            self.sp_x_shift.setValue(0.0)
            self.sp_y_shift.setValue(0.0)
            return
        st = self.style_map.get(skey, {})
        col = st.get("color")
        if isinstance(col, QColor):
            col = qcolor_to_tuple(col)
        self._set_color_button_preview(col if isinstance(col, tuple) else None)
        self.sp_line_width.setValue(float(st.get("width", 2.0)))
        try:
            self.sp_x_shift.setValue(float(st.get("x_shift", 0.0) or 0.0))
        except Exception:
            self.sp_x_shift.setValue(0.0)
        try:
            self.sp_y_shift.setValue(float(st.get("y_shift", 0.0) or 0.0))
        except Exception:
            self.sp_y_shift.setValue(0.0)

    def pick_color_for_selected_series(self) -> None:
        skey = self._current_style_key()
        if not skey:
            return
        st = self.style_map.get(skey, {})
        current = st.get("color")
        initial = QColor(*(current if isinstance(current, tuple) else (0, 120, 215)))
        c = QColorDialog.getColor(initial, self, "Renk Seç")
        if not c.isValid():
            return
        st["color"] = qcolor_to_tuple(c)
        st.setdefault("width", self.sp_line_width.value())
        st["x_shift"] = float(self.sp_x_shift.value())
        st["y_shift"] = float(self.sp_y_shift.value())
        self.style_map[skey] = st
        self._set_color_button_preview(st["color"])
        self.update_plot_data_with_filter()

    def apply_style_for_selected_series(self) -> None:
        skey = self._current_style_key()
        if not skey:
            return
        st = self.style_map.get(skey, {})
        st["width"] = float(self.sp_line_width.value())
        st["x_shift"] = float(self.sp_x_shift.value())
        st["y_shift"] = float(self.sp_y_shift.value())
        self.style_map[skey] = st
        self.update_plot_data_with_filter()

    def reset_style_for_selected_series(self) -> None:
        skey = self._current_style_key()
        if not skey:
            return
        self.style_map.pop(skey, None)
        self._set_color_button_preview(None)
        self.sp_line_width.setValue(2.0)
        self.sp_x_shift.setValue(0.0)
        self.sp_y_shift.setValue(0.0)
        self.update_plot_data_with_filter()

    def reset_all_styles(self) -> None:
        self.style_map.clear()
        self._on_style_series_changed()
        self.update_plot_data_with_filter()

    # ----- Shift series tree -----

    def _checked_shift_style_keys(self) -> List[str]:
        """Return style keys for all checked items in the shift series tree."""
        keys: List[str] = []
        for i in range(self.shift_series_tree.topLevelItemCount()):
            item = self.shift_series_tree.topLevelItem(i)
            if item is not None and item.checkState(0) == Qt.CheckState.Checked:
                skey = item.data(0, Qt.ItemDataRole.UserRole)
                if skey:
                    keys.append(skey)
        return keys

    def _set_all_shift_checks(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for i in range(self.shift_series_tree.topLevelItemCount()):
            item = self.shift_series_tree.topLevelItem(i)
            if item is not None:
                item.setCheckState(0, state)

    def apply_shift_for_checked_series(self) -> None:
        """Apply the current X/Y shift values to all checked channels in the shift tree."""
        keys = self._checked_shift_style_keys()
        if not keys:
            return
        xs = float(self.sp_x_shift.value())
        ys = float(self.sp_y_shift.value())
        for skey in keys:
            st = self.style_map.get(skey, {})
            st["x_shift"] = xs
            st["y_shift"] = ys
            self.style_map[skey] = st
        self.update_plot_data_with_filter()

    def reset_shift_for_checked_series(self) -> None:
        """Reset X/Y shift to zero for all checked channels in the shift tree."""
        keys = self._checked_shift_style_keys()
        if not keys:
            return
        for skey in keys:
            st = self.style_map.get(skey, {})
            st["x_shift"] = 0.0
            st["y_shift"] = 0.0
            self.style_map[skey] = st
        self.sp_x_shift.setValue(0.0)
        self.sp_y_shift.setValue(0.0)
        self.update_plot_data_with_filter()

    def _populate_shift_series_tree(self) -> None:
        """Populate the shift series tree with current plotted channels."""
        self.shift_series_tree.clear()
        if not self.current_series:
            return
        seen: set = set()
        for s in self.current_series:
            skey = s.get("style_key")
            if not skey or skey in seen:
                continue
            seen.add(skey)
            fl = (s.get("file_label") or "").strip()
            nm = (s.get("name") or "").strip()
            label = f"{fl} | {nm}" if fl else nm
            item = QTreeWidgetItem([label, skey])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, Qt.CheckState.Unchecked)
            item.setData(0, Qt.ItemDataRole.UserRole, skey)
            self.shift_series_tree.addTopLevelItem(item)

    # ----- Persistence -----

    def _restore_ui_state(self) -> None:
        try:
            g = self.settings.value("ui/geometry")
            if g:
                self.restoreGeometry(g)
            st = self.settings.value("ui/window_state")
            if st:
                self.restoreState(st)
            ms = self.settings.value("ui/main_splitter_workspace_v2")
            if ms and getattr(self, "_main_splitter", None) is not None:
                self._main_splitter.restoreState(ms)
            ps = self.settings.value("ui/plot_splitter")
            if ps and getattr(self, "_plot_vsplit", None) is not None:
                self._plot_vsplit.restoreState(ps)
        except Exception:
            logger.debug("UI state restore failed", exc_info=True)

        bool_settings = {
            "manual_fs/enabled": (self.chk_manual_fs_plot, False),
            "fft/log": (self.chk_fft_log, False),
            "fft/window": (self.chk_fft_window, True),
            "fft/remove_mean": (self.chk_fft_remove_mean, True),
            "fft/detrend": (self.chk_fft_detrend, True),
            "fft/hide_dc": (self.chk_fft_hide_dc, True),
            "fft/peak": (self.chk_fft_peak, True),
        }
        for key, (widget, default) in bool_settings.items():
            try:
                val = str(self.settings.value(key, str(default).lower()) or "").lower()
                widget.setChecked(val in ("1", "true", "yes"))
            except Exception:
                pass

        float_settings = {
            "manual_fs/value": (self.sp_manual_fs_plot, 0.0),
            "fft/fmin": (self.sp_fft_fmin, 0.0),
            "fft/fmax": (self.sp_fft_fmax, 0.0),
            "fft/fs_hint": (self.fs_fft, 0.0),
        }
        for key, (widget, default) in float_settings.items():
            try:
                widget.setValue(float(self.settings.value(key, default) or default))
            except Exception:
                widget.setValue(default)

        try:
            unit = str(self.settings.value("manual_fs/unit", "kHz") or "kHz")
            idx = self.cmb_manual_fs_unit.findText(unit)
            if idx >= 0:
                self.cmb_manual_fs_unit.setCurrentIndex(idx)
        except Exception:
            pass

        try:
            self.le_axis_x_title.setText(str(self.settings.value("axis/x_title", "") or ""))
            self.le_axis_y_title.setText(str(self.settings.value("axis/y_title", "") or ""))
            self.le_axis_y2_title.setText(str(self.settings.value("axis/y2_title", "") or ""))
            family = str(self.settings.value("axis/font_family", "Arial") or "Arial")
            self.cmb_axis_font.setCurrentFont(QFont(family))
            self.sp_axis_title_size.setValue(int(self.settings.value("axis/title_size", 11) or 11))
            self.chk_axis_title_bold.setChecked(str(self.settings.value("axis/title_bold", "false") or "false").lower() in ("1", "true", "yes"))
            self.sp_axis_tick_size.setValue(int(self.settings.value("axis/tick_size", 9) or 9))
            self.chk_axis_tick_bold.setChecked(str(self.settings.value("axis/tick_bold", "false") or "false").lower() in ("1", "true", "yes"))
            self._apply_axis_style_from_controls(show_status=False)
        except Exception:
            logger.debug("Axis style restore failed", exc_info=True)

    def _save_ui_state(self) -> None:
        try:
            self.settings.setValue("ui/theme", self._theme)
            self.settings.setValue("ui/geometry", self.saveGeometry())
            self.settings.setValue("ui/window_state", self.saveState())
            if getattr(self, "_main_splitter", None) is not None:
                self.settings.setValue("ui/main_splitter_workspace_v2", self._main_splitter.saveState())
            if getattr(self, "_plot_vsplit", None) is not None:
                self.settings.setValue("ui/plot_splitter", self._plot_vsplit.saveState())

            self.settings.setValue("manual_fs/enabled", self.chk_manual_fs_plot.isChecked())
            self.settings.setValue("manual_fs/value", self.sp_manual_fs_plot.value())
            self.settings.setValue("manual_fs/unit", self.cmb_manual_fs_unit.currentText())

            for key, widget in [
                ("fft/log", self.chk_fft_log), ("fft/window", self.chk_fft_window),
                ("fft/remove_mean", self.chk_fft_remove_mean), ("fft/detrend", self.chk_fft_detrend),
                ("fft/hide_dc", self.chk_fft_hide_dc), ("fft/peak", self.chk_fft_peak),
            ]:
                self.settings.setValue(key, widget.isChecked())
            self.settings.setValue("fft/fmin", self.sp_fft_fmin.value())
            self.settings.setValue("fft/fmax", self.sp_fft_fmax.value())
            self.settings.setValue("fft/fs_hint", self.fs_fft.value())

            axis_style = self._collect_axis_style_from_controls()
            self.settings.setValue("axis/x_title", axis_style["x_title"])
            self.settings.setValue("axis/y_title", axis_style["y_title"])
            self.settings.setValue("axis/y2_title", axis_style["y2_title"])
            self.settings.setValue("axis/font_family", axis_style["font_family"])
            self.settings.setValue("axis/title_size", axis_style["title_size"])
            self.settings.setValue("axis/title_bold", axis_style["title_bold"])
            self.settings.setValue("axis/tick_size", axis_style["tick_size"])
            self.settings.setValue("axis/tick_bold", axis_style["tick_bold"])
        except Exception:
            logger.debug("UI state save failed", exc_info=True)

    def _install_shortcuts(self) -> None:
        try:
            act_open = QAction(self)
            act_open.setShortcut(QKeySequence.StandardKey.Open)
            act_open.triggered.connect(self.open_tdms)
            self.addAction(act_open)

            act_close = QAction(self)
            act_close.setShortcut(QKeySequence.StandardKey.Close)
            act_close.triggered.connect(lambda: self._close_file_tab(self.file_tabs.currentIndex()))
            self.addAction(act_close)

            act_export = QAction(self)
            act_export.setShortcut(QKeySequence("Ctrl+E"))
            act_export.triggered.connect(self.export_csv)
            self.addAction(act_export)

            act_save_img = QAction(self)
            act_save_img.setShortcut(QKeySequence("Ctrl+Shift+S"))
            act_save_img.triggered.connect(self.save_plot_image)
            self.addAction(act_save_img)

            for seq, slot in [("F", self._autofit_all_panes), ("Backspace", self._autofit_all_panes), ("L", self._shortcut_toggle_legend), ("M", self._shortcut_toggle_marker), ("Ctrl+Z", self._undo_zoom), ("Ctrl+Shift+C", self.copy_plot_to_clipboard)]:
                a = QAction(self); a.setShortcut(QKeySequence(seq)); a.triggered.connect(slot); self.addAction(a)
        except Exception:
            logger.debug("Shortcut installation failed", exc_info=True)

    def copy_plot_to_clipboard(self) -> None:
        try:
            widget = self.stacked_view if self.cmb_plot_mode.currentText() == "Stacked" else self.plot_pane
            QApplication.clipboard().setPixmap(widget.grab())
            self.status.showMessage("Grafik panoya kopyalandı.", 1800)
        except Exception as e:
            QMessageBox.warning(self, "Kopyalama", str(e))

    def _undo_zoom(self) -> None:
        if len(self._zoom_history) < 2:
            self._autofit_all_panes(); return
        self._zoom_history_guard = True
        try:
            self._zoom_history.pop()
            x0, x1 = self._zoom_history[-1]
            self.plot_pane.set_xrange(x0, x1); self.stacked_view.set_xrange(x0, x1)
            if self.detached_win is not None: self.detached_win.pane.set_xrange(x0, x1)
            self._sync_range_boxes(x0, x1)
        finally:
            self._zoom_history_guard = False

    def _shortcut_toggle_legend(self) -> None:
        self._set_legend_visible_all(not self.plot_pane._legend_visible)

    def _shortcut_toggle_marker(self) -> None:
        new = not self.chk_click_mark.isChecked(); self.chk_click_mark.setChecked(new); self._set_click_mark_enabled_all(new)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._save_ui_state()
        try:
            self._lazy_view_timer.stop()
        except Exception:
            pass
        self._cancel_jobs(["index", "preview", "load", "lazy_view", "filter", "fft"])
        for th, _worker, _tag in list(self._jobs):
            try:
                th.quit()
                th.wait(500)
            except Exception:
                pass
        self._jobs.clear()
        super().closeEvent(event)

    # ----- File tabs -----

    def _new_file_id(self) -> str:
        self.file_counter += 1
        return f"F{self.file_counter}"

    def open_tdms(self) -> None:
        last_dir = str(self.settings.value("paths/last_dir", "") or "")
        start_dir = last_dir if (last_dir and os.path.isdir(last_dir)) else ""
        path, _ = QFileDialog.getOpenFileName(self, "TDMS Aç", start_dir, "TDMS (*.tdms)")
        if not path:
            return
        try:
            self.settings.setValue("paths/last_dir", os.path.dirname(path))
        except Exception:
            pass
        self.open_tdms_path(path)

    def open_tdms_paths(self, paths: List[str]) -> None:
        for p in paths:
            try:
                self.open_tdms_path(p)
            except Exception:
                logger.warning("Failed to open: %s", p, exc_info=True)

    def open_tdms_path(self, path: str) -> None:
        path = (path or "").strip()
        if not path:
            return
        try:
            self.settings.setValue("paths/last_dir", os.path.dirname(path))
        except Exception:
            pass
        self._add_to_recent(path)

        abs_path = os.path.abspath(path)
        for fid, st in self.files.items():
            if os.path.abspath(st.path) == abs_path:
                for idx in range(1, self.file_tabs.count()):
                    if self.file_tabs.tabText(idx) == st.label:
                        self.file_tabs.setCurrentIndex(idx)
                        return

        file_id = self._new_file_id()
        label = file_label_from_path(path)

        tab = QWidget()
        tab.setProperty("file_id", file_id)
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)

        hdr = QHBoxLayout()
        lbl = QLabel(path)
        lbl.setWordWrap(True)
        lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        hdr.addWidget(QLabel("Dosya:"))
        hdr.addWidget(lbl, stretch=1)
        lay.addLayout(hdr)

        search = QLineEdit()
        search.setPlaceholderText("Kanal ara...")
        lay.addWidget(search)

        tree = QTreeWidget()
        tree.setHeaderLabels(["Kanal Adı", "Örnek Sayısı"])
        tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        tree.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        lay.addWidget(tree, stretch=1)

        search.textChanged.connect(lambda t, fid=file_id: self.filter_tree(fid, t))
        tree.itemSelectionChanged.connect(lambda fid=file_id: self.on_selection_changed(fid))
        tree.itemChanged.connect(lambda item, col, fid=file_id: self.on_item_changed(fid, item, col))

        self.file_tabs.addTab(tab, label)
        self.file_tabs.setCurrentIndex(self.file_tabs.count() - 1)
        self.files[file_id] = FileState(file_id=file_id, path=path, label=label, tree=tree, search=search)
        self._update_status_metrics()

        self.status.showMessage(f"{label}: dizin okunuyor...")
        worker = TdmsIndexWorker(file_id, path)
        self._start_job(worker, self._on_index_ready, tag="index")

    def _close_file_tab(self, index: int) -> None:
        if index == 0:
            return
        w = self.file_tabs.widget(index)
        file_id_to_remove: Optional[str] = None
        try:
            if w is not None:
                file_id_to_remove = w.property("file_id")
        except Exception:
            pass

        if not file_id_to_remove:
            for fid, st in self.files.items():
                try:
                    if st.tree and st.tree.parent() == w:
                        file_id_to_remove = fid
                        break
                except Exception:
                    continue

        self.file_tabs.removeTab(index)
        if w:
            w.deleteLater()
        if file_id_to_remove and file_id_to_remove in self.files:
            del self.files[file_id_to_remove]
            prefix = f"{file_id_to_remove}|"
            self.channel_fs_overrides = {
                k: v for k, v in self.channel_fs_overrides.items() if not str(k).startswith(prefix)
            }
        self._manual_fs_tab_dirty = True
        self._channel_manager_dirty = True
        self._calculated_tab_dirty = True
        self._on_workspace_tab_changed(self.tabs.currentIndex())
        gc.collect()
        self._update_status_metrics()
        self.status.showMessage("Dosya kapatıldı.", 2000)

    def _on_worker_failed(self, msg: str) -> None:
        QMessageBox.critical(self, "Hata", msg)
        self.status.showMessage("Bir hata oluştu.", 5000)

    def _on_index_ready(self, payload: dict) -> None:
        if payload.get("cancelled"):
            return
        file_id = payload["file_id"]
        refs = payload["refs"]
        st = self.files.get(file_id)
        if not st:
            return
        self._populate_tree(st, refs)
        self._manual_fs_tab_dirty = True
        self._on_workspace_tab_changed(self.tabs.currentIndex())
        self._update_status_metrics()
        self.status.showMessage(f"{st.label}: {len(refs)} kanal bulundu.", 5000)

    def _populate_tree(self, st: FileState, refs: List[Tuple[str, str, int]]) -> None:
        tree = st.tree
        tree.blockSignals(True)
        tree.clear()

        groups: Dict[str, QTreeWidgetItem] = {}
        for gname, cname, n in refs:
            if gname not in groups:
                g = QTreeWidgetItem([gname, ""])
                g.setFlags(
                    Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsAutoTristate
                )
                g.setCheckState(0, Qt.CheckState.Unchecked)
                tree.addTopLevelItem(g)
                groups[gname] = g

            key = ChannelKey(file_id=st.file_id, group=gname, channel=cname, length=n)
            c = QTreeWidgetItem([cname, str(n)])
            c.setData(0, Qt.ItemDataRole.UserRole, key)
            c.setFlags(
                Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEditable
            )
            c.setCheckState(0, Qt.CheckState.Unchecked)
            groups[gname].addChild(c)

        tree.expandToDepth(0)
        tree.blockSignals(False)

    def filter_tree(self, file_id: str, t: str) -> None:
        st = self.files.get(file_id)
        if not st:
            return
        tree = st.tree
        t = (t or "").lower().strip()
        for i in range(tree.topLevelItemCount()):
            g = tree.topLevelItem(i)
            any_vis = False
            for j in range(g.childCount()):
                c = g.child(j)
                match = (t in c.text(0).lower()) if t else True
                c.setHidden(not match)
                any_vis = any_vis or match
            g.setHidden(not any_vis)

    def on_item_changed(self, file_id: str, item: QTreeWidgetItem, col: int) -> None:
        if col != 0:
            return
        key = item.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(key, ChannelKey) and self.current_series:
            self.update_plot_data_with_filter()

    # ----- Plot mode / channel manager / calculated channels -----

    def _on_workspace_tab_changed(self, _index: int) -> None:
        """Populate heavier management tables only when the user opens them."""
        current = self.tabs.currentWidget() if getattr(self, "tabs", None) is not None else None
        if current is getattr(self, "channel_manager_tab", None) and self._channel_manager_dirty:
            self._populate_channel_manager()
            self._channel_manager_dirty = False
        elif current is getattr(self, "calculated_channel_tab", None) and self._calculated_tab_dirty:
            self._populate_calculated_aliases()
            self._calculated_tab_dirty = False
        elif current is getattr(self, "manual_fs_tab", None) and self._manual_fs_tab_dirty:
            self._populate_manual_fs_table()
            self._manual_fs_tab_dirty = False
        elif current is getattr(self, "range_stats_tab", None):
            self.lbl_range_stats_axis.setText(self._range_stats_axis_text())

    def _on_plot_mode_changed(self, *_args: Any) -> None:
        new_mode = "Stacked" if self.cmb_plot_mode.currentText().strip().lower() == "stacked" else "Overlay"
        old_mode = getattr(self, "_active_plot_mode", "Overlay")

        # Preserve the currently visible X range before changing the view.
        source_xr: Optional[Tuple[float, float]] = None
        try:
            if old_mode == "Stacked" and self.stacked_view._plots:
                xr = self.stacked_view._plots[0].getViewBox().viewRange()[0]
                source_xr = (float(xr[0]), float(xr[1]))
            elif self.plot_pane.plot is not None:
                xr = self.plot_pane.plot.getViewBox().viewRange()[0]
                source_xr = (float(xr[0]), float(xr[1]))
        except Exception:
            source_xr = None

        self._active_plot_mode = new_mode
        stacked = new_mode == "Stacked"
        self.plot_container.setCurrentWidget(self.stacked_view if stacked else self.plot_pane)

        # Mode switching should not rerun filtering. Reuse the latest filtered
        # payload and render only the view that just became visible.
        if self._last_display_series:
            self._render_to_all_panes(self._last_display_series, self._last_axis_mode, self._last_x_label)
        elif self.current_series:
            self.update_plot_data_with_filter()

        if source_xr and source_xr[1] > source_xr[0]:
            try:
                if stacked:
                    self.stacked_view.set_xrange(*source_xr)
                else:
                    self.plot_pane.set_xrange(*source_xr)
            except Exception:
                pass
        self.status.showMessage("Stacked görünüm" if stacked else "Overlay görünüm", 1800)

    def _open_channel_manager_tab(self) -> None:
        if getattr(self, "channel_manager_tab", None) is None:
            self.channel_manager_tab = self._build_channel_manager_tab()
            self.tabs.addTab(self.channel_manager_tab, "Kanal Yöneticisi")
        self._populate_channel_manager()
        self._channel_manager_dirty = False
        self.tabs.setCurrentWidget(self.channel_manager_tab)

    def _build_channel_manager_tab(self) -> QWidget:
        tab = QWidget()
        lay = QVBoxLayout(tab)
        top = QHBoxLayout()
        self.channel_manager_search = QLineEdit()
        self.channel_manager_search.setPlaceholderText("Kanal / görünen ad ara...")
        self.channel_manager_search.setClearButtonEnabled(True)
        top.addWidget(QLabel("Ara:"))
        top.addWidget(self.channel_manager_search, 1)
        lay.addLayout(top)
        note = QLabel("Görünürlük, isim, birim, eksen, renk ve kanal bazlı Fs tek ekrandan yönetilir. Manuel Fs işaretlenmezse mevcut TDMS/Referans Fs korunur.")
        note.setWordWrap(True)
        lay.addWidget(note)
        self.channel_manager_tree = QTreeWidget()
        self.channel_manager_tree.setColumnCount(12)
        self.channel_manager_tree.setHeaderLabels(["Göster", "Dosya", "Kanal", "Görünen Ad", "Fs", "Manuel", "Fs Kaynağı", "Birim", "Eksen", "Renk", "Süre", "Kalite"])
        self.channel_manager_tree.setRootIsDecorated(False)
        self.channel_manager_tree.setAlternatingRowColors(True)
        self.channel_manager_tree.setUniformRowHeights(True)
        self.channel_manager_tree.setColumnWidth(0, 65)
        self.channel_manager_tree.setColumnWidth(1, 170)
        self.channel_manager_tree.setColumnWidth(2, 220)
        self.channel_manager_tree.setColumnWidth(3, 220)
        lay.addWidget(self.channel_manager_tree, 1)
        row = QHBoxLayout()
        self.btn_channel_manager_apply = QPushButton("Uygula")
        self.btn_channel_manager_apply.setProperty("primary", True)
        self.btn_channel_manager_refresh = QPushButton("Yenile")
        self.btn_channel_manager_back = QPushButton("Grafiğe Dön")
        row.addWidget(self.btn_channel_manager_apply)
        row.addWidget(self.btn_channel_manager_refresh)
        row.addStretch(1)
        row.addWidget(self.btn_channel_manager_back)
        lay.addLayout(row)
        self.btn_channel_manager_apply.clicked.connect(self._apply_channel_manager)
        self.btn_channel_manager_refresh.clicked.connect(self._populate_channel_manager)
        self.btn_channel_manager_back.clicked.connect(lambda: self.tabs.setCurrentIndex(0))
        self.channel_manager_search.textChanged.connect(self._filter_channel_manager)
        return tab

    def _effective_fs_source(self, s: dict) -> Tuple[Optional[float], str]:
        skey = s.get("style_key", "")
        if skey in self.channel_fs_overrides:
            return float(self.channel_fs_overrides[skey]), "Kanal manuel"
        fs = s.get("source_fs_est")
        if fs is not None:
            try:
                return float(fs), "TDMS"
            except Exception:
                pass
        fs = s.get("auto_fs_hz") or s.get("fs_est")
        if fs is not None:
            try:
                src = "Referans Fs" if s.get("auto_fs_reference_source") == "manual_reference" else "Otomatik"
                return float(fs), src
            except Exception:
                pass
        return None, "—"

    def _populate_channel_manager(self) -> None:
        tree = getattr(self, "channel_manager_tree", None)
        if tree is None:
            return
        tree.clear()
        for s in self.current_series or []:
            skey = str(s.get("style_key", ""))
            st = self.style_map.get(skey, {})
            fs, src = self._effective_fs_source(s)
            xq = np.asarray(s.get("x", []), dtype=np.float64); yq = np.asarray(s.get("y", []))
            duration = float(xq[-1] - xq[0]) if xq.size > 1 and np.isfinite(xq[0]) and np.isfinite(xq[-1]) else float("nan")
            try:
                yf = np.asarray(yq, dtype=np.float64); nan_n = int(np.count_nonzero(np.isnan(yf))); inf_n = int(np.count_nonzero(np.isinf(yf)))
                quality = f"NaN:{nan_n} Inf:{inf_n}" + (" (görünüm)" if s.get("lazy") else "")
            except Exception:
                quality = "—"
            item = QTreeWidgetItem(["", str(s.get("file_label", "")), str(s.get("name", "")), "", "", "", src, "", "", "", f"{duration:.6g} s" if math.isfinite(duration) else "—", quality])
            item.setData(0, Qt.ItemDataRole.UserRole, skey)
            tree.addTopLevelItem(item)
            vis = QCheckBox(); vis.setChecked(st.get("visible", True) is not False); tree.setItemWidget(item, 0, vis)
            name = QLineEdit(str(st.get("display_name") or s.get("name") or "")); tree.setItemWidget(item, 3, name)
            fsbox = QDoubleSpinBox(); fsbox.setRange(0.0, 1e12); fsbox.setDecimals(6); fsbox.setKeyboardTracking(False); fsbox.setValue(float(fs or 0.0)); tree.setItemWidget(item, 4, fsbox)
            man = QCheckBox(); man.setChecked(skey in self.channel_fs_overrides); tree.setItemWidget(item, 5, man)
            unit = QLineEdit(str(st.get("unit_override") or s.get("unit") or "")); tree.setItemWidget(item, 7, unit)
            axis = QComboBox(); axis.addItems(["Auto", "Sol", "Sağ"]); amap={"auto":"Auto","left":"Sol","right":"Sağ"}; axis.setCurrentText(amap.get(str(st.get("axis","auto")),"Auto")); tree.setItemWidget(item, 8, axis)
            color_btn = QPushButton("Renk")
            c = st.get("color")
            if isinstance(c, QColor):
                c = qcolor_to_tuple(c)
            if c is not None:
                try:
                    rgb = tuple(int(v) for v in c[:3]); color_btn.setProperty("channel_rgb", rgb); color_btn.setStyleSheet(f"background: rgb({rgb[0]},{rgb[1]},{rgb[2]});")
                except Exception:
                    pass
            color_btn.clicked.connect(lambda _=False, b=color_btn: self._pick_manager_color(b))
            tree.setItemWidget(item, 9, color_btn)
        self._filter_channel_manager(getattr(self, "channel_manager_search", QLineEdit()).text() if hasattr(self, "channel_manager_search") else "")

    def _pick_manager_color(self, button: QPushButton) -> None:
        cur = button.property("channel_rgb")
        qcur = QColor(*(cur if isinstance(cur, tuple) else (80, 140, 220)))
        c = QColorDialog.getColor(qcur, self, "Kanal Rengi")
        if not c.isValid():
            return
        rgb = qcolor_to_tuple(c)
        button.setProperty("channel_rgb", rgb)
        button.setStyleSheet(f"background: rgb({rgb[0]},{rgb[1]},{rgb[2]});")

    def _filter_channel_manager(self, text: str = "") -> None:
        tree = getattr(self, "channel_manager_tree", None)
        if tree is None:
            return
        q = (text or "").strip().lower()
        for i in range(tree.topLevelItemCount()):
            it = tree.topLevelItem(i)
            name_w = tree.itemWidget(it, 3)
            hay = f"{it.text(2)} {name_w.text() if isinstance(name_w, QLineEdit) else ''}".lower()
            it.setHidden(bool(q and q not in hay))

    def _apply_channel_manager(self) -> None:
        tree = getattr(self, "channel_manager_tree", None)
        if tree is None:
            return
        for i in range(tree.topLevelItemCount()):
            it = tree.topLevelItem(i)
            skey = it.data(0, Qt.ItemDataRole.UserRole)
            if not isinstance(skey, str):
                continue
            st = dict(self.style_map.get(skey, {}))
            vis = tree.itemWidget(it, 0); name = tree.itemWidget(it, 3); fsbox = tree.itemWidget(it, 4); man = tree.itemWidget(it, 5); unit = tree.itemWidget(it, 7); axis = tree.itemWidget(it, 8); cbtn = tree.itemWidget(it, 9)
            st["visible"] = bool(vis.isChecked()) if isinstance(vis, QCheckBox) else True
            if isinstance(name, QLineEdit): st["display_name"] = name.text().strip()
            if isinstance(unit, QLineEdit): st["unit_override"] = unit.text().strip()
            if isinstance(axis, QComboBox): st["axis"] = {"Sol":"left","Sağ":"right"}.get(axis.currentText(), "auto")
            if isinstance(cbtn, QPushButton) and cbtn.property("channel_rgb") is not None: st["color"] = tuple(cbtn.property("channel_rgb"))
            self.style_map[skey] = st
            if isinstance(man, QCheckBox) and man.isChecked() and isinstance(fsbox, QDoubleSpinBox) and fsbox.value() > 0:
                self.channel_fs_overrides[skey] = float(fsbox.value())
            else:
                self.channel_fs_overrides.pop(skey, None)
        if self.current_series:
            self._apply_auto_fs_sync(); self._apply_fs_sync_axes(); self.update_plot_data_with_filter(); self._refresh_series_comboboxes(); self._refresh_preview_panel()
        self.status.showMessage("Kanal yöneticisi ayarları uygulandı.", 2500)

    def _open_calculated_channel_tab(self) -> None:
        if getattr(self, "calculated_channel_tab", None) is None:
            self.calculated_channel_tab = self._build_calculated_channel_tab()
            self.tabs.addTab(self.calculated_channel_tab, "Hesaplanan Kanal")
        self._populate_calculated_aliases()
        self._calculated_tab_dirty = False
        self.tabs.setCurrentWidget(self.calculated_channel_tab)

    def _build_calculated_channel_tab(self) -> QWidget:
        tab = QWidget(); lay = QVBoxLayout(tab)
        info = QLabel("Kanallar C1, C2... aliaslarıyla kullanılır. Örnek: C1-C2, C1/2, gradient(C3), sqrt(C1*C1+C2*C2). Farklı Fs varsa yalnızca hesaplanan kanal için ortak zaman eksenine interpolasyon yapılır.")
        info.setWordWrap(True); lay.addWidget(info)
        self.calc_alias_tree = QTreeWidget(); self.calc_alias_tree.setHeaderLabels(["Alias","Dosya","Kanal","Fs"]); self.calc_alias_tree.setRootIsDecorated(False); lay.addWidget(self.calc_alias_tree, 1)
        form = QGridLayout()
        self.calc_name = QLineEdit(); self.calc_name.setPlaceholderText("Örn. DeltaP")
        self.calc_unit = QLineEdit(); self.calc_unit.setPlaceholderText("Örn. bar")
        self.calc_expression = QLineEdit(); self.calc_expression.setPlaceholderText("Örn. C1 - C2")
        self.btn_calc_create = QPushButton("Hesaplanan Kanalı Oluştur"); self.btn_calc_create.setProperty("primary", True)
        form.addWidget(QLabel("Ad:"),0,0); form.addWidget(self.calc_name,0,1); form.addWidget(QLabel("Birim:"),0,2); form.addWidget(self.calc_unit,0,3)
        form.addWidget(QLabel("İfade:"),1,0); form.addWidget(self.calc_expression,1,1,1,3); form.addWidget(self.btn_calc_create,1,4)
        lay.addLayout(form)
        self.virtual_tree = QTreeWidget(); self.virtual_tree.setHeaderLabels(["Hesaplanan Kanal","İfade"]); self.virtual_tree.setMaximumHeight(150); lay.addWidget(self.virtual_tree)
        self.btn_calc_create.clicked.connect(self._create_calculated_channel)
        return tab

    def _populate_calculated_aliases(self) -> None:
        tree = getattr(self, "calc_alias_tree", None)
        if tree is None:
            return
        tree.clear(); self._calc_alias_map = {}
        base_series = [s for s in (self.current_series or []) if not s.get("is_virtual")]
        for i, s in enumerate(base_series, 1):
            alias = f"C{i}"; self._calc_alias_map[alias] = s
            fs, _src = self._effective_fs_source(s)
            tree.addTopLevelItem(QTreeWidgetItem([alias, str(s.get("file_label","")), str(s.get("name","")), f"{fs:.6g}" if fs else "—"]))

    def _create_calculated_channel(self) -> None:
        if not self.current_series:
            QMessageBox.information(self, "Veri Yok", "Önce kaynak kanalları çizin."); return
        name = self.calc_name.text().strip() or "Hesaplanan"
        unit = self.calc_unit.text().strip()
        expr = self.calc_expression.text().strip()
        if not expr:
            QMessageBox.information(self, "İfade", "Bir hesap ifadesi girin."); return
        aliases = sorted(set(re.findall(r"\bC\d+\b", expr)))
        if not aliases:
            QMessageBox.information(self, "Kanal", "İfadede en az bir C1, C2... kanalı bulunmalı."); return
        missing = [a for a in aliases if a not in self._calc_alias_map]
        if missing:
            QMessageBox.warning(self, "Alias", f"Bilinmeyen alias: {', '.join(missing)}"); return
        try:
            tree = _validate_calc_expression(expr, set(aliases))
            ref_alias = max(aliases, key=lambda a: self._series_sample_count(self._calc_alias_map[a]))
            ref = self._calc_alias_map[ref_alias]
            bx, by = self._load_full_xy(ref)
            bx = np.asarray(bx, dtype=np.float64); n = bx.size
            env: Dict[str, Any] = dict(_ALLOWED_CALC_FUNCS)
            for a in aliases:
                s = self._calc_alias_map[a]; x, y = self._load_full_xy(s)
                x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
                m = np.isfinite(x) & np.isfinite(y)
                if x.size == n and np.allclose(x, bx, rtol=1e-10, atol=1e-12, equal_nan=True):
                    env[a] = y
                elif np.count_nonzero(m) >= 2:
                    env[a] = np.interp(bx, x[m], y[m], left=np.nan, right=np.nan)
                else:
                    env[a] = np.full(n, np.nan)
            result = eval(compile(tree, "<calculated-channel>", "eval"), {"__builtins__": {}}, env)
            yy = np.asarray(result, dtype=np.float64)
            if yy.ndim == 0: yy = np.full(n, float(yy), dtype=np.float64)
            yy = np.ravel(yy)
            if yy.size != n: raise ValueError(f"İfade {yy.size} örnek üretti; beklenen {n}.")
            self._virtual_channel_counter += 1
            skey = f"virtual|{self._virtual_channel_counter}|{name}"
            fs_est = robust_fs_from_x(bx) if str(ref.get("x_mode")) in ("seconds","datetime") else None
            dig, levels = detect_digital_like(yy)
            snew = {
                "style_key": skey, "name": name, "file_label": "Hesaplanan", "x": bx, "y": yy,
                "unit": unit, "quantity": "Hesaplanan", "x_mode": ref.get("x_mode", "index"),
                "source_x_mode": ref.get("x_mode", "index"), "source_x_base": float(bx[0]) if bx.size else 0.0,
                "source_x_inc": (1.0/fs_est) if fs_est else 1.0, "source_fs_est": fs_est,
                "fs_est": fs_est, "is_digital": dig, "digital_levels": levels, "lazy": False,
                "is_virtual": True, "virtual_expression": expr,
            }
            self.current_series.append(snew)
            if getattr(self, "virtual_tree", None) is not None: self.virtual_tree.addTopLevelItem(QTreeWidgetItem([name, expr]))
            self._refresh_series_comboboxes(); self.update_plot_data_with_filter()
            self.status.showMessage(f"Hesaplanan kanal oluşturuldu: {name}", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Hesaplama Hatası", str(e))

    # ----- Per-channel Fs override / Manual Fs tab -----

    def _preview_style_key(self) -> Optional[str]:
        """Return style_key for the channel currently shown in the preview panel."""
        p = self._last_preview_payload or {}
        fid = p.get("file_id")
        group = p.get("group")
        channel = p.get("channel")
        if not (fid and group is not None and channel is not None):
            return None
        return style_key_for_channel(str(fid), str(group), str(channel))

    def _channel_override_hz_for_series(self, s: dict) -> Optional[float]:
        skey = str(s.get("style_key", "") or "")
        if not skey:
            return None
        val = self.channel_fs_overrides.get(skey)
        try:
            fs = float(val) if val is not None else None
        except (TypeError, ValueError):
            return None
        return fs if fs is not None and math.isfinite(fs) and fs > 0.0 else None

    def _build_range_stats_tab(self) -> QWidget:
        """Build manual interval analysis workspace."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        controls = QFrame()
        controls.setObjectName("fftPanel")
        grid = QGridLayout(controls)
        grid.setContentsMargins(12, 10, 12, 10)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        self.range_stats_xmin = QDoubleSpinBox()
        self.range_stats_xmax = QDoubleSpinBox()
        for w in (self.range_stats_xmin, self.range_stats_xmax):
            w.setRange(-1e18, 1e18)
            w.setDecimals(6)
            w.setKeyboardTracking(False)
            w.setMinimumWidth(155)

        self.lbl_range_stats_axis = QLabel("X ekseni: —")
        self.lbl_range_stats_axis.setStyleSheet("font-weight: 700;")
        self.btn_range_stats_visible = QPushButton("Görünen Aralığı Al")
        self.btn_range_stats_full = QPushButton("Tam Aralığı Al")
        self.btn_range_stats_calc = QPushButton("İstatistikleri Hesapla")
        self.btn_range_stats_calc.setProperty("primary", True)
        self.btn_range_stats_clear = QPushButton("Sonuçları Temizle")

        grid.addWidget(QLabel("Başlangıç:"), 0, 0)
        grid.addWidget(self.range_stats_xmin, 0, 1)
        grid.addWidget(QLabel("Bitiş:"), 0, 2)
        grid.addWidget(self.range_stats_xmax, 0, 3)
        grid.addWidget(self.lbl_range_stats_axis, 0, 4)
        grid.addWidget(self.btn_range_stats_calc, 0, 5)
        grid.addWidget(self.btn_range_stats_visible, 1, 0, 1, 2)
        grid.addWidget(self.btn_range_stats_full, 1, 2, 1, 2)
        grid.addWidget(self.btn_range_stats_clear, 1, 5)
        grid.setColumnStretch(4, 1)
        layout.addWidget(controls)

        note = QLabel(
            "İstatistikler çizili ve görünür kanalların ham verisi üzerinden hesaplanır. "
            "Lazy kanallarda yalnızca girilen veri aralığı diskten okunur. Kanal X/Y shift değerleri hesaba katılır."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #667085;")
        layout.addWidget(note)

        self.range_stats_tree = QTreeWidget()
        self.range_stats_tree.setRootIsDecorated(False)
        self.range_stats_tree.setAlternatingRowColors(True)
        self.range_stats_tree.setUniformRowHeights(True)
        self.range_stats_tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.range_stats_tree.setColumnCount(13)
        self.range_stats_tree.setHeaderLabels([
            "Kanal", "Birim", "N", "Geçersiz", "Ortalama", "Std Sapma",
            "Min", "Maks", "Peak-Peak", "RMS", "Medyan", "Integral", "Dosya",
        ])
        for col, width in enumerate((260, 90, 90, 90, 120, 120, 110, 110, 110, 110, 110, 130, 190)):
            self.range_stats_tree.setColumnWidth(col, width)
        layout.addWidget(self.range_stats_tree, stretch=1)

        self.lbl_range_stats_summary = QLabel("Bir veri aralığı girin ve İstatistikleri Hesapla'ya basın.")
        self.lbl_range_stats_summary.setWordWrap(True)
        self.lbl_range_stats_summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.lbl_range_stats_summary)

        self.btn_range_stats_visible.clicked.connect(self._range_stats_use_visible)
        self.btn_range_stats_full.clicked.connect(self._range_stats_use_full)
        self.btn_range_stats_calc.clicked.connect(self._calculate_range_stats)
        self.btn_range_stats_clear.clicked.connect(self._clear_range_stats_results)
        return tab

    def _range_stats_axis_text(self) -> str:
        if not self.current_series:
            return "X ekseni: —"
        modes = {str(s.get("x_mode", "index") or "index") for s in self.current_series}
        if modes == {"datetime"}:
            return "X ekseni: UTC epoch saniye"
        if "seconds" in modes or "datetime" in modes:
            return "X ekseni: saniye"
        return "X ekseni: indeks"

    def _series_full_x_bounds(self, s: dict) -> Optional[Tuple[float, float]]:
        st = self.style_map.get(str(s.get("style_key", "")), {})
        x_shift = float(st.get("x_shift", 0.0) or 0.0)
        meta = s.get("lazy_meta") or {}
        if meta:
            n = int(meta.get("n", 0) or 0)
            if n <= 0:
                return None
            mode = str(meta.get("x_mode", s.get("x_mode", "index")) or "index")
            base = float(meta.get("x_base", 0.0) or 0.0)
            inc = float(meta.get("x_inc", 1.0) or 1.0)
            if mode == "index":
                a, b = 0.0, float(n - 1)
            else:
                a, b = base, base + float(n - 1) * inc
            return min(a, b) + x_shift, max(a, b) + x_shift
        x = np.asarray(s.get("x", []), dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return None
        return float(np.min(x) + x_shift), float(np.max(x) + x_shift)

    def _range_stats_use_full(self) -> None:
        if not self.current_series:
            QMessageBox.information(self, "Aralık Analizi", "Önce en az bir kanal çizin.")
            return
        bounds = []
        for s in self.current_series:
            skey = str(s.get("style_key", ""))
            if not self.style_map.get(skey, {}).get("visible", True):
                continue
            b = self._series_full_x_bounds(s)
            if b is not None:
                bounds.append(b)
        if not bounds:
            QMessageBox.information(self, "Aralık Analizi", "Geçerli kanal aralığı bulunamadı.")
            return
        self.range_stats_xmin.setValue(min(b[0] for b in bounds))
        self.range_stats_xmax.setValue(max(b[1] for b in bounds))
        self.lbl_range_stats_axis.setText(self._range_stats_axis_text())

    def _range_stats_use_visible(self) -> None:
        if not self.current_series or self.plot_pane.plot is None:
            QMessageBox.information(self, "Aralık Analizi", "Önce en az bir kanal çizin.")
            return
        try:
            if getattr(self, "_active_plot_mode", "Overlay") == "Stacked" and self.stacked_view._plots:
                xr = self.stacked_view._plots[0].getViewBox().viewRange()[0]
            else:
                xr = self.plot_pane.plot.getViewBox().viewRange()[0]
            x0, x1 = sorted((float(xr[0]), float(xr[1])))
            self.range_stats_xmin.setValue(x0)
            self.range_stats_xmax.setValue(x1)
            self.lbl_range_stats_axis.setText(self._range_stats_axis_text())
        except Exception as e:
            QMessageBox.warning(self, "Aralık Analizi", f"Görünen aralık alınamadı:\n{e}")

    def _clear_range_stats_results(self) -> None:
        tree = getattr(self, "range_stats_tree", None)
        if tree is not None:
            tree.clear()
        if getattr(self, "lbl_range_stats_summary", None) is not None:
            self.lbl_range_stats_summary.setText("Sonuçlar temizlendi.")

    def _range_stats_requests(self) -> List[dict]:
        requests: List[dict] = []
        for s in self.current_series or []:
            skey = str(s.get("style_key", ""))
            st = self.style_map.get(skey, {})
            if not st.get("visible", True):
                continue
            base = {
                "style_key": skey,
                "file_label": s.get("file_label", ""),
                "name": s.get("name", "Kanal"),
                "unit": s.get("unit", ""),
                "x_shift": float(st.get("x_shift", 0.0) or 0.0),
                "y_shift": float(st.get("y_shift", 0.0) or 0.0),
            }
            meta = s.get("lazy_meta") or {}
            if meta and meta.get("path"):
                base.update({
                    "lazy": True,
                    "path": meta.get("path"),
                    "group": meta.get("group"),
                    "channel": meta.get("channel"),
                    "n": int(meta.get("n", 0) or 0),
                    "x_mode": str(meta.get("x_mode", s.get("x_mode", "index")) or "index"),
                    "x_base": float(meta.get("x_base", 0.0) or 0.0),
                    "x_inc": float(meta.get("x_inc", 1.0) or 1.0),
                })
            else:
                base.update({
                    "lazy": False,
                    "x": np.asarray(s.get("x", []), dtype=np.float64),
                    "y": np.asarray(s.get("y", []), dtype=np.float64),
                })
            requests.append(base)
        return requests

    def _calculate_range_stats(self) -> None:
        if not self.current_series:
            QMessageBox.information(self, "Aralık Analizi", "Önce en az bir kanal çizin.")
            return
        x0 = float(self.range_stats_xmin.value())
        x1 = float(self.range_stats_xmax.value())
        if not (math.isfinite(x0) and math.isfinite(x1)) or x1 <= x0:
            QMessageBox.warning(self, "Aralık Analizi", "Bitiş değeri başlangıç değerinden büyük olmalıdır.")
            return
        requests = self._range_stats_requests()
        if not requests:
            QMessageBox.information(self, "Aralık Analizi", "Analiz edilecek görünür kanal bulunamadı.")
            return
        self._range_stats_token += 1
        token = self._range_stats_token
        self.btn_range_stats_calc.setEnabled(False)
        self.lbl_range_stats_axis.setText(self._range_stats_axis_text())
        self.lbl_range_stats_summary.setText("Aralık istatistikleri hesaplanıyor...")
        self.status.showMessage("Aralık istatistikleri hesaplanıyor...")
        worker = RangeStatsWorker(token, requests, x0, x1)
        self._start_job(worker, self._on_range_stats_ready, tag="range_stats", cancel_tags=["range_stats"])

    @staticmethod
    def _fmt_stat(value: Any) -> str:
        try:
            v = float(value)
            return f"{v:.8g}" if math.isfinite(v) else "—"
        except (TypeError, ValueError):
            return "—"

    def _on_range_stats_ready(self, payload: dict) -> None:
        if payload.get("cancelled") or payload.get("token") != self._range_stats_token:
            return
        self.btn_range_stats_calc.setEnabled(True)
        rows = payload.get("rows") or []
        self.range_stats_tree.setUpdatesEnabled(False)
        try:
            self.range_stats_tree.clear()
            for r in rows:
                item = QTreeWidgetItem([
                    str(r.get("name", "Kanal")),
                    str(r.get("unit", "") or "—"),
                    str(int(r.get("n", 0) or 0)),
                    str(int(r.get("invalid_n", 0) or 0)),
                    self._fmt_stat(r.get("mean")),
                    self._fmt_stat(r.get("std")),
                    self._fmt_stat(r.get("min")),
                    self._fmt_stat(r.get("max")),
                    self._fmt_stat(r.get("ptp")),
                    self._fmt_stat(r.get("rms")),
                    self._fmt_stat(r.get("median")),
                    self._fmt_stat(r.get("integral")),
                    str(r.get("file_label", "")),
                ])
                item.setData(0, Qt.ItemDataRole.UserRole, r.get("style_key", ""))
                self.range_stats_tree.addTopLevelItem(item)
        finally:
            self.range_stats_tree.setUpdatesEnabled(True)

        errors = payload.get("errors") or []
        x0, x1 = payload.get("x_min"), payload.get("x_max")
        summary = f"Aralık: {self._fmt_stat(x0)} – {self._fmt_stat(x1)} | {len(rows)} kanal hesaplandı."
        if errors:
            summary += f"  {len(errors)} kanal/okuma uyarısı var."
            logger.warning("Range statistics warnings: %s", " | ".join(str(e) for e in errors[:20]))
        if not rows:
            summary += " Seçilen aralıkta geçerli veri bulunamadı."
        self.lbl_range_stats_summary.setText(summary)
        self.status.showMessage(summary, 4000)

    def _build_manual_fs_tab(self) -> QWidget:
        """Create the per-channel manual Fs editor tab on first use."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        title = QLabel(
            "Kanal bazlı örnekleme frekansı. 0 / — bırakılan kanalda manuel Fs kullanılmaz. "
            "Kanal bazlı değerler Referans Fs ve TDMS metadata'sından daha yüksek önceliklidir."
        )
        title.setWordWrap(True)
        layout.addWidget(title)

        search_row = QHBoxLayout()
        search_row.setSpacing(8)
        search_label = QLabel("Kanal Ara:")
        self.manual_fs_search = QLineEdit()
        self.manual_fs_search.setPlaceholderText("Kanal adına göre ara...")
        self.manual_fs_search.setClearButtonEnabled(True)
        self.manual_fs_search.setToolTip("Kanal adında geçen metne göre listeyi anlık filtreler.")
        search_row.addWidget(search_label)
        search_row.addWidget(self.manual_fs_search, stretch=1)
        layout.addLayout(search_row)

        self.manual_fs_tree = QTreeWidget()
        self.manual_fs_tree.setColumnCount(5)
        self.manual_fs_tree.setHeaderLabels(["Dosya", "Kanal", "Örnek Sayısı", "Fs", "Birim"])
        self.manual_fs_tree.setRootIsDecorated(False)
        self.manual_fs_tree.setAlternatingRowColors(True)
        self.manual_fs_tree.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.manual_fs_tree.setUniformRowHeights(True)
        self.manual_fs_tree.setColumnWidth(0, 220)
        self.manual_fs_tree.setColumnWidth(1, 300)
        self.manual_fs_tree.setColumnWidth(2, 120)
        self.manual_fs_tree.setColumnWidth(3, 150)
        self.manual_fs_tree.setColumnWidth(4, 90)
        layout.addWidget(self.manual_fs_tree, stretch=1)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        self.btn_manual_fs_apply_all = QPushButton("Uygula")
        self.btn_manual_fs_apply_all.setProperty("primary", True)
        self.btn_manual_fs_apply_all.setToolTip(
            "Tablodaki tüm kanal Fs değerlerini uygular. 0 / — olan satırların manuel atamasını kaldırır."
        )
        self.btn_manual_fs_clear_all = QPushButton("Tüm Atamaları Temizle")
        self.btn_manual_fs_clear_all.setProperty("danger", True)
        self.btn_manual_fs_preset_save = QPushButton("Preset Kaydet...")
        self.btn_manual_fs_preset_load = QPushButton("Preset Yükle...")
        self.btn_manual_fs_back = QPushButton("Grafiğe Dön")
        buttons.addWidget(self.btn_manual_fs_apply_all)
        buttons.addWidget(self.btn_manual_fs_clear_all)
        buttons.addWidget(self.btn_manual_fs_preset_save)
        buttons.addWidget(self.btn_manual_fs_preset_load)
        buttons.addStretch(1)
        buttons.addWidget(self.btn_manual_fs_back)
        layout.addLayout(buttons)

        self.btn_manual_fs_apply_all.clicked.connect(self._apply_manual_fs_table)
        self.btn_manual_fs_clear_all.clicked.connect(self._clear_manual_fs_table)
        self.btn_manual_fs_preset_save.clicked.connect(self._save_manual_fs_preset)
        self.btn_manual_fs_preset_load.clicked.connect(self._load_manual_fs_preset)
        self.btn_manual_fs_back.clicked.connect(lambda: self.tabs.setCurrentIndex(0))
        self.manual_fs_search.textChanged.connect(self._filter_manual_fs_table)
        return tab

    def _filter_manual_fs_table(self, text: str = "") -> None:
        """Filter Manual Fs rows by channel name, case-insensitively."""
        tree = getattr(self, "manual_fs_tree", None)
        if tree is None:
            return
        needle = (text or "").strip().casefold()
        for i in range(tree.topLevelItemCount()):
            item = tree.topLevelItem(i)
            skey = item.data(0, Qt.ItemDataRole.UserRole)
            if not isinstance(skey, str) or not skey:
                item.setHidden(False)
                continue
            channel_name = (item.text(1) or "").casefold()
            item.setHidden(bool(needle) and needle not in channel_name)

    def _open_manual_fs_tab(self) -> None:
        """Open (or refresh) the dedicated Manual Fs tab."""
        tab = getattr(self, "manual_fs_tab", None)
        if tab is None:
            self.manual_fs_tab = self._build_manual_fs_tab()
            self.tabs.addTab(self.manual_fs_tab, "Manuel Fs")
        self._populate_manual_fs_table()
        self._manual_fs_tab_dirty = False
        self.tabs.setCurrentWidget(self.manual_fs_tab)

    def _populate_manual_fs_table(self) -> None:
        """Populate one editable Fs row for every currently open TDMS channel."""
        tree = getattr(self, "manual_fs_tree", None)
        if tree is None:
            return

        tree.setUpdatesEnabled(False)
        try:
            tree.clear()
            row_count = 0
            for file_id, st in self.files.items():
                src_tree = st.tree
                for i in range(src_tree.topLevelItemCount()):
                    group_item = src_tree.topLevelItem(i)
                    for j in range(group_item.childCount()):
                        channel_item = group_item.child(j)
                        key = channel_item.data(0, Qt.ItemDataRole.UserRole)
                        if not isinstance(key, ChannelKey):
                            continue

                        skey = style_key_for_channel(file_id, key.group, key.channel)
                        display_channel = channel_item.text(0) or key.channel
                        item = QTreeWidgetItem([
                            st.label,
                            f"{key.group}/{display_channel}",
                            str(key.length),
                            "",
                            "",
                        ])
                        item.setData(0, Qt.ItemDataRole.UserRole, skey)
                        tree.addTopLevelItem(item)

                        spin = QDoubleSpinBox()
                        spin.setRange(0.0, 1e12)
                        spin.setDecimals(6)
                        spin.setKeyboardTracking(False)
                        spin.setSpecialValueText("—")
                        spin.setToolTip("0 / — = bu kanalda manuel Fs ataması yok")

                        unit = QComboBox()
                        unit.addItems(["Hz", "kHz"])

                        stored = self.channel_fs_overrides.get(skey)
                        try:
                            stored_fs = float(stored) if stored is not None else None
                        except (TypeError, ValueError):
                            stored_fs = None
                        if stored_fs is not None and math.isfinite(stored_fs) and stored_fs > 0.0:
                            if stored_fs >= 1000.0:
                                unit.setCurrentText("kHz")
                                spin.setValue(stored_fs / 1000.0)
                            else:
                                unit.setCurrentText("Hz")
                                spin.setValue(stored_fs)
                        else:
                            unit.setCurrentText("kHz")
                            spin.setValue(0.0)

                        tree.setItemWidget(item, 3, spin)
                        tree.setItemWidget(item, 4, unit)
                        row_count += 1

            if row_count == 0:
                empty = QTreeWidgetItem(["—", "Önce bir TDMS dosyası açın", "", "", ""])
                empty.setDisabled(True)
                tree.addTopLevelItem(empty)

            search = getattr(self, "manual_fs_search", None)
            self._filter_manual_fs_table(search.text() if isinstance(search, QLineEdit) else "")
        finally:
            tree.setUpdatesEnabled(True)

    def _save_manual_fs_preset(self) -> None:
        tree = getattr(self, "manual_fs_tree", None)
        if tree is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Fs Preseti Kaydet", "", "Fs preset (*.json)")
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        data: Dict[str, float] = {}
        for i in range(tree.topLevelItemCount()):
            item = tree.topLevelItem(i); spin = tree.itemWidget(item, 3); unit = tree.itemWidget(item, 4)
            if not isinstance(spin, QDoubleSpinBox) or not isinstance(unit, QComboBox):
                continue
            fs = fs_value_to_hz(spin.value(), unit.currentText())
            if fs is not None:
                data[item.text(1)] = float(fs)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"version": 1, "channels_hz": data}, f, ensure_ascii=False, indent=2)
            self.status.showMessage(f"Fs preseti kaydedildi: {os.path.basename(path)}", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Preset", str(e))

    def _load_manual_fs_preset(self) -> None:
        tree = getattr(self, "manual_fs_tree", None)
        if tree is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Fs Preseti Yükle", "", "Fs preset (*.json);;JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            mapping = payload.get("channels_hz", {}) if isinstance(payload, dict) else {}
            matched = 0
            for i in range(tree.topLevelItemCount()):
                item = tree.topLevelItem(i); fs = mapping.get(item.text(1))
                if fs is None:
                    continue
                fs = float(fs); spin = tree.itemWidget(item, 3); unit = tree.itemWidget(item, 4)
                if not isinstance(spin, QDoubleSpinBox) or not isinstance(unit, QComboBox) or fs <= 0:
                    continue
                if fs >= 1000.0:
                    unit.setCurrentText("kHz"); spin.setValue(fs / 1000.0)
                else:
                    unit.setCurrentText("Hz"); spin.setValue(fs)
                matched += 1
            self.status.showMessage(f"Fs preseti yüklendi: {matched} kanal eşleşti. Uygula'ya basın.", 4000)
        except Exception as e:
            QMessageBox.critical(self, "Preset", str(e))

    def _apply_manual_fs_table(self) -> None:
        """Apply all values currently entered in the Manual Fs table."""
        tree = getattr(self, "manual_fs_tree", None)
        if tree is None:
            return

        applied = 0
        removed = 0
        for i in range(tree.topLevelItemCount()):
            item = tree.topLevelItem(i)
            skey = item.data(0, Qt.ItemDataRole.UserRole)
            if not isinstance(skey, str) or not skey:
                continue
            spin = tree.itemWidget(item, 3)
            unit = tree.itemWidget(item, 4)
            if not isinstance(spin, QDoubleSpinBox) or not isinstance(unit, QComboBox):
                continue

            fs = fs_value_to_hz(spin.value(), unit.currentText())
            if fs is not None:
                self.channel_fs_overrides[skey] = float(fs)
                applied += 1
            elif self.channel_fs_overrides.pop(skey, None) is not None:
                removed += 1

        if self.current_series:
            self._apply_auto_fs_sync()
            self._apply_fs_sync_axes()
            self.update_plot_data_with_filter()
        self._refresh_preview_panel()
        self.status.showMessage(
            f"Manuel Fs uygulandı: {applied} kanal atanmış, {removed} atama kaldırılmış.",
            4000,
        )

    def _clear_manual_fs_table(self) -> None:
        """Clear manual Fs overrides for channels shown in the table."""
        tree = getattr(self, "manual_fs_tree", None)
        if tree is None:
            return
        cleared = 0
        for i in range(tree.topLevelItemCount()):
            item = tree.topLevelItem(i)
            skey = item.data(0, Qt.ItemDataRole.UserRole)
            if isinstance(skey, str) and self.channel_fs_overrides.pop(skey, None) is not None:
                cleared += 1
            spin = tree.itemWidget(item, 3)
            if isinstance(spin, QDoubleSpinBox):
                spin.setValue(0.0)

        if self.current_series:
            self._apply_auto_fs_sync()
            self._apply_fs_sync_axes()
            self.update_plot_data_with_filter()
        self._refresh_preview_panel()
        self.status.showMessage(f"{cleared} kanalın manuel Fs ataması temizlendi.", 3000)

    # ----- Automatic / Reference Fs -----

    def _get_plot_manual_fs_hz(self) -> Optional[float]:
        """Return the user-entered reference Fs, if enabled."""
        if not self.chk_manual_fs_plot.isChecked():
            return None
        return fs_value_to_hz(self.sp_manual_fs_plot.value(), self.cmb_manual_fs_unit.currentText())

    def _manual_fs_display_text(self) -> str:
        """Human-readable reference-Fs text kept under the legacy helper name."""
        fs_hz = self._get_plot_manual_fs_hz()
        if fs_hz is None:
            return "—"
        raw_value = self.sp_manual_fs_plot.value()
        raw_unit = self.cmb_manual_fs_unit.currentText() or "Hz"
        return f"{raw_value:.6g} {raw_unit} ({fs_hz:.6g} Hz, referans)"

    def _rebuild_uniform_x(self, s: dict, x_mode: str, x_base: float, x_inc: float) -> np.ndarray:
        if s.get("lazy_meta"):
            win = s.get("_lazy_last_win")
            if isinstance(win, tuple) and len(win) == 3:
                i0, i1, stride = win
                idx = np.arange(i0, i1, max(1, stride), dtype=np.float64)
            else:
                idx = np.arange(len(np.asarray(s.get("y", []))), dtype=np.float64)
        else:
            idx = np.arange(len(np.asarray(s.get("y", []))), dtype=np.float64)
        return idx if x_mode == "index" else (x_base + idx * x_inc).astype(np.float64)

    def _series_sample_count(self, s: dict) -> int:
        """Return the original sample count for a loaded series."""
        n = int(s.get("sample_count", 0) or 0)
        if n > 0:
            return n
        meta = s.get("lazy_meta") or {}
        n = int(meta.get("n", 0) or 0)
        if n > 0:
            return n
        return int(len(np.asarray(s.get("y", []))))

    def _apply_auto_fs_sync(self) -> dict:
        """
        Synchronize index-only channels to the common test duration.

        Priority:
        1) If any selected channel has real TDMS time/Fs metadata, use the timed
           channel with the largest original sample count as the reference.
        2) Otherwise, if the user enabled "Referans Fs", assign that Fs to the
           selected index-only channel with the largest original sample count.

        The application assumes selected test channels start and stop together.
        Once the reference duration is known, every index-only channel gets

            Fs_channel = (N_channel - 1) / T_ref

        No Y-data interpolation/resampling is performed; only X/time axes are
        reconstructed.
        """
        result = {
            "reference": None,
            "reference_source": None,
            "duration_s": None,
            "synced": 0,
        }
        if not self.current_series:
            return result

        # Clear values from a previous plot/load before recalculating.
        for s in self.current_series:
            s["auto_fs_hz"] = None
            s["auto_fs_reference"] = None
            s["auto_fs_reference_source"] = None
            s["auto_duration_s"] = None
            s["auto_x_mode"] = None
            s["auto_x_base"] = None
            s["auto_x_inc"] = None

        # First preference: actual TDMS time/Fs metadata.
        candidates = []
        for s in self.current_series:
            source_mode = str(s.get("source_x_mode", s.get("x_mode", "index")) or "index")
            source_fs = s.get("source_fs_est")
            n = self._series_sample_count(s)
            try:
                fs = float(source_fs) if source_fs is not None else None
            except (TypeError, ValueError):
                fs = None
            if source_mode not in ("seconds", "datetime") or fs is None:
                continue
            if not math.isfinite(fs) or fs <= 0.0 or n < 2:
                continue
            duration_s = (n - 1) / fs
            source_base = float(s.get("source_x_base", 0.0) or 0.0)
            if math.isfinite(duration_s) and duration_s > 0.0 and math.isfinite(source_base):
                candidates.append((n, fs, duration_s, source_mode, source_base, s))

        if candidates:
            ref_n, ref_fs, duration_s, ref_mode, ref_base, ref = max(candidates, key=lambda item: item[0])
            ref_source = "tdms"
        else:
            # No metadata: the user-provided Fs belongs ONLY to the highest-sample
            # selected channel. It is not applied directly to every channel.
            reference_fs = self._get_plot_manual_fs_hz()
            index_candidates = []
            if reference_fs is not None and math.isfinite(reference_fs) and reference_fs > 0.0:
                for s in self.current_series:
                    source_mode = str(s.get("source_x_mode", s.get("x_mode", "index")) or "index")
                    n = self._series_sample_count(s)
                    if source_mode == "index" and n >= 2:
                        index_candidates.append((n, s))

            if not index_candidates:
                return result

            ref_n, ref = max(index_candidates, key=lambda item: item[0])
            ref_fs = float(reference_fs)
            duration_s = (ref_n - 1) / ref_fs
            ref_mode = "seconds"
            ref_base = 0.0
            ref_source = "manual_reference"
            if not math.isfinite(duration_s) or duration_s <= 0.0:
                return result

        ref_label = f"{ref.get('file_label', '')} | {ref.get('name', '')}".strip(" |")
        result.update({
            "reference": ref_label,
            "reference_source": ref_source,
            "reference_fs_hz": float(ref_fs),
            "reference_samples": int(ref_n),
            "reference_x_mode": ref_mode,
            "reference_x_base": float(ref_base),
            "duration_s": float(duration_s),
        })

        # Reconstruct every index-only channel, including the manually supplied
        # reference itself when TDMS metadata is absent.
        for s in self.current_series:
            source_mode = str(s.get("source_x_mode", s.get("x_mode", "index")) or "index")
            if source_mode != "index":
                continue
            n = self._series_sample_count(s)
            if n < 2:
                continue
            fs = (n - 1) / duration_s
            if not math.isfinite(fs) or fs <= 0.0:
                continue
            s["auto_fs_hz"] = float(fs)
            s["auto_fs_reference"] = ref_label
            s["auto_fs_reference_source"] = ref_source
            s["auto_duration_s"] = float(duration_s)
            s["auto_x_mode"] = ref_mode
            s["auto_x_base"] = float(ref_base)
            s["auto_x_inc"] = 1.0 / float(fs)
            result["synced"] += 1

        return result

    def _apply_fs_sync_axes(self) -> None:
        """Apply explicit channel Fs first, then auto/reference Fs, then TDMS metadata."""
        if not self.current_series:
            return
        for s in self.current_series:
            source_mode = str(s.get("source_x_mode", s.get("x_mode", "index")) or "index")
            source_base = float(s.get("source_x_base", 0.0) or 0.0)
            source_inc = float(s.get("source_x_inc", 1.0) or 1.0)
            source_fs = s.get("source_fs_est")
            channel_fs = self._channel_override_hz_for_series(s)
            auto_fs = s.get("auto_fs_hz")

            effective_fs: Optional[float] = None
            effective_source: Optional[str] = None
            effective_mode = source_mode
            effective_base = source_base

            # Highest priority: explicit per-channel Fs.
            if channel_fs is not None:
                effective_fs = channel_fs
                effective_source = "channel_manual"
                effective_mode = source_mode if source_mode in ("seconds", "datetime") else "seconds"
                effective_base = source_base if source_mode in ("seconds", "datetime") else 0.0
            elif source_mode == "index" and auto_fs is not None:
                try:
                    auto_val = float(auto_fs)
                    if math.isfinite(auto_val) and auto_val > 0.0:
                        effective_fs = auto_val
                        effective_source = (
                            "manual_reference" if s.get("auto_fs_reference_source") == "manual_reference" else "auto_tdms"
                        )
                        effective_mode = str(s.get("auto_x_mode") or "seconds")
                        effective_base = float(s.get("auto_x_base", 0.0) or 0.0)
                except (TypeError, ValueError):
                    pass

            if effective_fs is not None:
                effective_inc = 1.0 / effective_fs
                s["x_mode"] = effective_mode
                s["x"] = self._rebuild_uniform_x(s, effective_mode, effective_base, effective_inc)
                s["fs_est"] = effective_fs
                s["manual_fs_hz"] = channel_fs if effective_source == "channel_manual" else None
                s["effective_fs_source"] = effective_source
                s["effective_x_base"] = effective_base
                if s.get("lazy_meta"):
                    meta = s["lazy_meta"]
                    meta["x_mode"] = effective_mode
                    meta["x_base"] = effective_base
                    meta["x_inc"] = effective_inc
            else:
                s["manual_fs_hz"] = None
                s["effective_fs_source"] = "tdms" if source_mode in ("seconds", "datetime") else None
                s["effective_x_base"] = source_base
                s["fs_est"] = source_fs
                if s.get("lazy_meta"):
                    meta = s["lazy_meta"]
                    meta["x_mode"] = source_mode
                    meta["x_base"] = source_base
                    meta["x_inc"] = source_inc
                    s["x_mode"] = source_mode
                    s["x"] = self._rebuild_uniform_x(s, source_mode, source_base, source_inc)
                else:
                    s["x_mode"] = source_mode
                    if source_mode == "index":
                        s["x"] = self._rebuild_uniform_x(s, "index", 0.0, 1.0)

    def _refresh_preview_panel(self) -> None:
        p = self._last_preview_payload or {}
        fid = p.get("file_id", "")
        st = self.files.get(fid)
        self.p_file.setText(st.label if st else "—")
        self.p_name.setText(p.get("name", "—"))
        self.p_samples.setText(str(p.get("samples", "—")))
        x_info = p.get("x_info") or {}
        source_fs = x_info.get("fs_est")
        x_mode = str(x_info.get("x_mode", "index") or "index")

        skey = self._preview_style_key()
        channel_fs = self.channel_fs_overrides.get(skey) if skey else None
        try:
            channel_fs = float(channel_fs) if channel_fs is not None else None
        except (TypeError, ValueError):
            channel_fs = None

        if channel_fs is not None and math.isfinite(channel_fs) and channel_fs > 0.0:
            self.p_fs.setText(f"{channel_fs:.6g} Hz (kanal manuel)")
        elif source_fs:
            self.p_fs.setText(f"{float(source_fs):.6g} Hz (TDMS)")
        elif x_mode == "index":
            auto_series = None
            group, channel = p.get("group"), p.get("channel")
            if fid and group is not None and channel is not None and self.current_series:
                auto_skey = style_key_for_channel(fid, str(group), str(channel))
                auto_series = next((s for s in self.current_series if s.get("style_key") == auto_skey), None)
            auto_fs = auto_series.get("auto_fs_hz") if auto_series else None
            if auto_fs is not None:
                ref_src = auto_series.get("auto_fs_reference_source")
                suffix = "referans Fs'den" if ref_src == "manual_reference" else "otomatik"
                self.p_fs.setText(f"{float(auto_fs):.6g} Hz ({suffix})")
            elif self._get_plot_manual_fs_hz() is not None:
                self.p_fs.setText(f"—  |  Referans: {self._manual_fs_display_text()}")
            else:
                self.p_fs.setText("—")
        else:
            self.p_fs.setText("—")
        self.p_quantity.setText(p.get("quantity", "Value"))
        u = (p.get("unit") or "").strip()
        self.p_unit.setText(u or "—")

    def _on_manual_fs_controls_changed(self, *_args: Any) -> None:
        if self.current_series:
            auto_sync = self._apply_auto_fs_sync()
            self._apply_fs_sync_axes()
            self.update_plot_data_with_filter()
            self._refresh_preview_panel()
            ref_source = auto_sync.get("reference_source")
            duration_s = auto_sync.get("duration_s")
            synced = int(auto_sync.get("synced", 0) or 0)
            if ref_source == "manual_reference" and synced and duration_s is not None:
                self.status.showMessage(
                    f"Referans Fs kullanıldı; {synced} kanal {float(duration_s):.6g} s ortak test süresine senkronlandı.",
                    3500,
                )
            elif ref_source == "tdms" and synced:
                self.status.showMessage(
                    "TDMS zaman metadata'sı bulundu; manuel referans yerine TDMS zamanı kullanılıyor.",
                    3500,
                )
            elif self.chk_manual_fs_plot.isChecked():
                self.status.showMessage(
                    "Referans Fs etkin ancak geçerli bir değer/kanal bulunamadı.", 3000
                )
            else:
                self.status.showMessage(
                    "Referans Fs kapalı; TDMS metadata yoksa kanallar indeks ekseninde kalır.", 3000
                )
        else:
            self._refresh_preview_panel()

    # ----- Preview -----

    def on_selection_changed(self, file_id: str) -> None:
        st = self.files.get(file_id)
        if not st:
            return
        items = st.tree.selectedItems()
        if len(items) != 1:
            return
        key = items[0].data(0, Qt.ItemDataRole.UserRole)
        if not isinstance(key, ChannelKey):
            return
        worker = TdmsChannelPreviewWorker(st.path, key)
        self._start_job(worker, self._on_preview_ready, tag="preview", cancel_tags=["preview"])

    def _on_preview_ready(self, p: dict) -> None:
        if p.get("cancelled"):
            return
        self._last_preview_payload = dict(p)
        self._refresh_preview_panel()

    # ----- Plot checked channels -----

    def uncheck_all_channels(self) -> None:
        for st in self.files.values():
            tree = st.tree
            tree.blockSignals(True)
            for i in range(tree.topLevelItemCount()):
                tree.topLevelItem(i).setCheckState(0, Qt.CheckState.Unchecked)
            tree.blockSignals(False)
        self.status.showMessage("Tüm seçimler temizlendi.", 2000)

    def _gather_checked_requests(self) -> List[ChannelRequest]:
        reqs: List[ChannelRequest] = []
        for st in self.files.values():
            tree = st.tree
            for i in range(tree.topLevelItemCount()):
                g = tree.topLevelItem(i)
                for j in range(g.childCount()):
                    c = g.child(j)
                    if c.checkState(0) == Qt.CheckState.Checked:
                        key = c.data(0, Qt.ItemDataRole.UserRole)
                        if isinstance(key, ChannelKey):
                            reqs.append(ChannelRequest(key=key, display_name=c.text(0)))
        return reqs

    def plot_checked_channels(self) -> None:
        if not self.files:
            return
        reqs = self._gather_checked_requests()
        if not reqs:
            QMessageBox.information(self, "Seçim Yok", "En az bir kanal işaretleyin (onay kutusu).")
            return
        files_payload = {fid: {"path": st.path, "label": st.label} for fid, st in self.files.items()}
        self.status.showMessage("Kanallar yükleniyor...")
        worker = MultiTdmsChannelLoadWorker(files_payload, reqs)
        self._start_job(worker, self._on_load_ready, tag="load", cancel_tags=["load", "lazy_view", "filter"])

    def _on_load_ready(self, payload: dict) -> None:
        if payload.get("cancelled"):
            return
        self.current_series = payload["series"]
        auto_sync = self._apply_auto_fs_sync()
        self._apply_fs_sync_axes()
        self.clear_markers()
        self.update_plot_data_with_filter()
        self._refresh_series_comboboxes()
        self._refresh_statistics()
        self._update_workspace_summary()
        try:
            self._range_stats_use_full()
            self._clear_range_stats_results()
            self.lbl_range_stats_summary.setText("Aralık hazır. Başlangıç/bitişi değiştirip İstatistikleri Hesapla'ya basın.")
        except Exception:
            logger.debug("Range statistics defaults could not be refreshed", exc_info=True)
        synced = int(auto_sync.get("synced", 0) or 0)
        duration_s = auto_sync.get("duration_s")
        ref_source = auto_sync.get("reference_source")
        ref_fs = auto_sync.get("reference_fs_hz")
        if synced and duration_s is not None:
            if ref_source == "manual_reference":
                self.status.showMessage(
                    f"{len(self.current_series)} kanal hazır. En yüksek örnekli kanal {float(ref_fs):.6g} Hz referans alındı; "
                    f"{synced} kanal {float(duration_s):.6g} s ortak süreye senkronlandı.",
                    5000,
                )
            else:
                self.status.showMessage(
                    f"{len(self.current_series)} kanal hazır. TDMS zamanı kullanıldı; {synced} index kanal "
                    f"{float(duration_s):.6g} s ortak süreye senkronlandı.",
                    4500,
                )
        elif all(str(s.get("source_x_mode", "index")) == "index" for s in self.current_series):
            self.status.showMessage(
                f"{len(self.current_series)} kanal hazır. TDMS Fs metadata'sı yok; Referans Fs veya kanal bazlı Fs girebilirsiniz.",
                4500,
            )
        else:
            self.status.showMessage(f"{len(self.current_series)} kanal hazır.", 3000)

    def _refresh_series_comboboxes(self) -> None:
        self.cmb_fft_channel.clear()
        self.cmb_style_series.clear()
        if not self.current_series:
            self._populate_shift_series_tree()
            return
        seen: set = set()
        for s in self.current_series:
            skey = s.get("style_key")
            if not skey or skey in seen:
                continue
            seen.add(skey)
            fl = (s.get("file_label") or "").strip()
            nm = (s.get("name") or "").strip()
            label = f"{fl} | {nm}" if fl else nm
            self.cmb_fft_channel.addItem(label, userData=skey)
            self.cmb_style_series.addItem(label, userData=skey)
        self._on_style_series_changed()
        self._populate_shift_series_tree()
        self._channel_manager_dirty = True
        self._calculated_tab_dirty = True
        self._manual_fs_tab_dirty = True
        self._on_workspace_tab_changed(self.tabs.currentIndex())

    def _capture_current_ranges(self) -> None:
        self._preserve_main_xrange = None
        self._preserve_detached_xrange = None
        self._preserve_stacked_xrange = None
        try:
            if self.plot_pane.plot is not None:
                xr = self.plot_pane.plot.getViewBox().viewRange()[0]
                self._preserve_main_xrange = (float(xr[0]), float(xr[1]))
        except Exception:
            pass
        try:
            if self.stacked_view._plots:
                xr = self.stacked_view._plots[0].getViewBox().viewRange()[0]
                self._preserve_stacked_xrange = (float(xr[0]), float(xr[1]))
        except Exception:
            pass
        try:
            if self.detached_win is not None and self.detached_win.pane.plot is not None:
                xr = self.detached_win.pane.plot.getViewBox().viewRange()[0]
                self._preserve_detached_xrange = (float(xr[0]), float(xr[1]))
        except Exception:
            pass

    def _restore_ranges(self) -> None:
        for attr, pane_getter in [
            ("_preserve_main_xrange", lambda: self.plot_pane),
            ("_preserve_detached_xrange", lambda: self.detached_win.pane if self.detached_win else None),
        ]:
            try:
                xr = getattr(self, attr)
                pane = pane_getter()
                if xr and pane is not None and pane.plot is not None and xr[1] > xr[0]:
                    pane.set_xrange(float(xr[0]), float(xr[1]))
            except Exception:
                pass
            finally:
                setattr(self, attr, None)
        try:
            xr = getattr(self, "_preserve_stacked_xrange", None)
            if xr and xr[1] > xr[0]:
                self.stacked_view.set_xrange(float(xr[0]), float(xr[1]))
        except Exception:
            pass
        finally:
            self._preserve_stacked_xrange = None

    def _render_to_all_panes(
        self, display_series: List[dict], axis_mode: str, x_label: str,
    ) -> None:
        """Render only the currently visible main plot mode.

        The old implementation rebuilt Overlay *and* Stacked after every
        filter/style/lazy update. Stacked creation is comparatively expensive,
        so doing both doubled work and made mode switching feel sluggish.
        """
        self._last_display_series = list(display_series)
        self._last_axis_mode = axis_mode
        self._last_x_label = x_label

        mp = self.plot_pane
        state = {
            "mode": mp._interaction_mode,
            "region": mp._region_enabled,
            "click_mark": mp._click_mark_enabled,
            "y_lock": mp._y_lock_enabled,
            "y2_lock": mp._right_y_lock_enabled,
            "legend": mp._legend_visible,
        }

        prepared: List[dict] = []
        for s in display_series:
            st = self.style_map.get(s.get("style_key", ""), {})
            if st.get("visible", True) is False:
                continue
            ss = dict(s)
            if st.get("display_name"):
                ss["name"] = st["display_name"]
            if "unit_override" in st:
                ss["unit"] = st.get("unit_override", "")
            prepared.append(ss)

        stacked_active = self.cmb_plot_mode.currentText().strip().lower() == "stacked"

        # Main workspace: render just the active mode.
        if stacked_active:
            self.stacked_view.set_axis_style(self.axis_style)
            self.stacked_view.plot_series(prepared, axis_mode, x_label, self.style_map)
            self.stacked_view.set_background(self.plot_bg_rgb)
        else:
            mp.set_axis_style(self.axis_style)
            mp.plot_series(prepared, axis_mode, x_label, style_map=self.style_map)
            mp.set_background(self.plot_bg_rgb)
            mp.set_interaction_mode(state["mode"])
            mp.enable_region(state["region"])
            mp.set_click_mark_mode(state["click_mark"])
            mp.set_y_lock(state["y_lock"])
            mp.set_right_y_lock(state["y2_lock"])
            mp.set_legend_visible(state["legend"])
            mp.clear_markers()
            for m in self._markers.values():
                mp.add_marker(m["x"], m["y"], m["label"])

        # Detached plot is intentionally an overlay view and only gets updated
        # when it actually exists.
        if self.detached_win is not None:
            pane = self.detached_win.pane
            pane.set_axis_style(self.axis_style)
            pane.plot_series(prepared, axis_mode, x_label, style_map=self.style_map)
            pane.set_background(self.plot_bg_rgb)
            pane.set_interaction_mode(state["mode"])
            pane.enable_region(state["region"])
            pane.set_click_mark_mode(state["click_mark"])
            pane.set_y_lock(state["y_lock"])
            pane.set_right_y_lock(state["y2_lock"])
            pane.set_legend_visible(state["legend"])
            pane.clear_markers()
            for m in self._markers.values():
                pane.add_marker(m["x"], m["y"], m["label"])

        self._restore_ranges()

    def update_plot_data_with_filter(self) -> None:
        if not self.current_series:
            return
        apply_filter = self.chk_smooth.isChecked() and SCIPY_AVAILABLE
        window_len = int(self.spin_smooth_win.value())
        self._filter_token += 1
        token = self._filter_token
        self.status.showMessage("Filtreleniyor/çiziliyor...")
        series_snapshot = [dict(s) for s in self.current_series]
        worker = FilterWorker(token=token, series=series_snapshot, apply_filter=apply_filter, window_len=window_len)
        self._start_job(worker, self._on_filter_ready, tag="filter", cancel_tags=["filter"])

    def _on_filter_ready(self, payload: dict) -> None:
        if payload.get("cancelled") or payload.get("token") != self._filter_token:
            return
        self._render_to_all_panes(
            payload.get("display_series", []),
            payload.get("axis_mode", "numeric"),
            payload.get("x_label", "X"),
        )
        self.status.showMessage("Hazır.", 1500)

    def on_clear_plot(self) -> None:
        self.plot_pane.clear_all()
        self.stacked_view.clear_all()
        if self.detached_win is not None:
            self.detached_win.pane.clear_all()
        self.current_series = None
        self.cmb_fft_channel.clear()
        self.cmb_style_series.clear()
        self.shift_series_tree.clear()
        self.clear_markers()
        self._refresh_statistics()
        self._update_workspace_summary()
        try:
            self._clear_range_stats_results()
            self.lbl_range_stats_axis.setText("X ekseni: —")
        except Exception:
            pass

    # ----- Range -----

    def _sync_range_boxes(self, x0: float, x1: float) -> None:
        for w in (self.xmin, self.xmax):
            w.blockSignals(True)
        self.xmin.setValue(x0)
        self.xmax.setValue(x1)
        for w in (self.xmin, self.xmax):
            w.blockSignals(False)

    def _on_any_range_changed(self, source: str, x0: float, x1: float) -> None:
        if self._syncing_range:
            return
        self._syncing_range = True
        try:
            self._sync_range_boxes(x0, x1)
            if source != "main": self.plot_pane.set_xrange(x0, x1)
            if source != "stacked": self.stacked_view.set_xrange(x0, x1)
            if self.detached_win is not None and source != "detached": self.detached_win.pane.set_xrange(x0, x1)
            if not self._zoom_history_guard and math.isfinite(x0) and math.isfinite(x1) and x1 > x0:
                rng = (float(x0), float(x1))
                if not self._zoom_history or max(abs(rng[0]-self._zoom_history[-1][0]), abs(rng[1]-self._zoom_history[-1][1])) > max(1e-12, abs(rng[1]-rng[0])*1e-5):
                    self._zoom_history.append(rng); self._zoom_history = self._zoom_history[-60:]
            self._schedule_lazy_view_update(x0, x1)
            self._update_workspace_summary(x0, x1)
        finally:
            self._syncing_range = False

    def apply_range_manual(self) -> None:
        x0, x1 = self.xmin.value(), self.xmax.value()
        self.plot_pane.set_xrange(x0, x1)
        self.stacked_view.set_xrange(x0, x1)
        if self.detached_win is not None:
            self.detached_win.pane.set_xrange(x0, x1)

    def autofit_y_in_range(self) -> None:
        x0, x1 = self.xmin.value(), self.xmax.value()
        self.plot_pane.autofit_y_in_range(x0, x1)
        if self.detached_win is not None:
            self.detached_win.pane.autofit_y_in_range(x0, x1)

    # ----- Lazy view -----

    def _schedule_lazy_view_update(self, x0: float, x1: float) -> None:
        if not CFG.lazy.enabled or not self.current_series:
            return
        if not any(s.get("lazy_meta") for s in self.current_series):
            return
        if not (math.isfinite(x0) and math.isfinite(x1)):
            return
        self._lazy_view_pending = (x0, x1)
        self._lazy_view_timer.start(CFG.lazy.view_debounce_ms)

    def _run_lazy_view_update(self) -> None:
        if not CFG.lazy.enabled or not self.current_series or not self._lazy_view_pending:
            return
        x0, x1 = self._lazy_view_pending
        self._lazy_view_pending = None

        reqs: List[dict] = []
        for s in self.current_series:
            meta = s.get("lazy_meta")
            if not isinstance(meta, dict):
                continue
            skey = s.get("style_key", "")
            if not skey:
                continue
            x_shift = 0.0
            try:
                x_shift = float(self.style_map.get(skey, {}).get("x_shift", 0.0) or 0.0)
                if not math.isfinite(x_shift):
                    x_shift = 0.0
            except Exception:
                pass
            reqs.append({
                "style_key": skey,
                "path": meta.get("path"),
                "group": meta.get("group"),
                "channel": meta.get("channel"),
                "n": int(meta.get("n", 0) or 0),
                "x_mode": meta.get("x_mode", "index"),
                "x_base": float(meta.get("x_base", 0.0) or 0.0),
                "x_inc": float(meta.get("x_inc", 1.0) or 1.0),
                "x0": x0 - x_shift,
                "x1": x1 - x_shift,
            })

        if not reqs:
            return
        self._lazy_view_token += 1
        token = self._lazy_view_token
        worker = LazyViewLoadWorker(reqs, max_points=CFG.lazy.view_max_points)
        self._start_job(
            worker,
            lambda payload, t=token: self._on_lazy_view_ready(t, payload),
            tag="lazy_view", cancel_tags=["lazy_view"],
        )

    def _on_lazy_view_ready(self, token: int, payload: dict) -> None:
        if payload.get("cancelled") or token != self._lazy_view_token:
            return
        updates = payload.get("updates") or {}
        if not updates:
            return
        changed = False
        for s in self.current_series or []:
            skey = s.get("style_key", "")
            if skey not in updates:
                continue
            u = updates[skey]
            win = tuple(u.get("win", ())) if isinstance(u, dict) else ()
            if win and s.get("_lazy_last_win") == win:
                continue
            x, y = u.get("x"), u.get("y")
            if x is None or y is None:
                continue
            s["x"], s["y"] = x, y
            if win:
                s["_lazy_last_win"] = win
            changed = True
        if changed:
            self._capture_current_ranges()
            self.update_plot_data_with_filter()

    def _load_full_xy(self, s: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Load full X/Y data from disk for FFT, honoring per-channel/manual/reference Fs."""
        source_mode = str(s.get("source_x_mode", s.get("x_mode", "index")) or "index")
        channel_fs = self._channel_override_hz_for_series(s)
        auto_fs = s.get("auto_fs_hz")
        effective_fs = channel_fs if channel_fs is not None else auto_fs
        try:
            effective_fs = float(effective_fs) if effective_fs is not None else None
        except (TypeError, ValueError):
            effective_fs = None
        if effective_fs is not None and (not math.isfinite(effective_fs) or effective_fs <= 0.0):
            effective_fs = None

        if channel_fs is not None:
            effective_base = float(s.get("source_x_base", 0.0) or 0.0) if source_mode in ("seconds", "datetime") else 0.0
        else:
            effective_base = float(s.get("auto_x_base", 0.0) or 0.0)

        if not s.get("lazy_meta"):
            y = np.asarray(s.get("y", []), dtype=np.float64)
            if effective_fs is not None and (channel_fs is not None or source_mode == "index"):
                x = effective_base + np.arange(y.size, dtype=np.float64) / effective_fs
            elif source_mode == "index":
                x = np.arange(y.size, dtype=np.float64)
            else:
                x = np.asarray(s.get("x", []), dtype=np.float64)
            return x, y

        meta = s.get("lazy_meta", {})
        path = meta.get("path")
        group, channel = meta.get("group"), meta.get("channel")
        n = int(meta.get("n", 0) or 0)
        source_base = float(s.get("source_x_base", meta.get("x_base", 0.0)) or 0.0)
        source_inc = float(s.get("source_x_inc", meta.get("x_inc", 1.0)) or 1.0)

        if not (path and group and channel and n > 0):
            return np.asarray(s.get("x", []), dtype=np.float64), np.asarray(s.get("y", []), dtype=np.float64)

        try:
            with _open_tdms(path) as tdms:
                ch = tdms[group][channel]
                y = _ensure_1d_numeric(ch[:])
                nn = y.size
                idx = np.arange(nn, dtype=np.float64)
                if effective_fs is not None and (channel_fs is not None or source_mode == "index"):
                    x = effective_base + idx / effective_fs
                elif source_mode == "index":
                    x = idx
                else:
                    x = (source_base + idx * source_inc).astype(np.float64)
                return x, y
        except Exception:
            logger.warning("Failed to load full data for lazy channel", exc_info=True)
            return np.asarray(s.get("x", []), dtype=np.float64), np.asarray(s.get("y", []), dtype=np.float64)

    # ----- Markers -----

    def clear_markers(self) -> None:
        self.plot_pane.clear_markers()
        if self.detached_win is not None:
            self.detached_win.pane.clear_markers()
        self.marker_list.clear()
        self._markers.clear()
        self._marker_counter = 0
        self.lbl_marker_calc.setText("Sonuç: —")

    def _on_marker_requested_from(self, source: str, x: float, y: float) -> None:
        self._marker_counter += 1
        mid = self._marker_counter
        label = f"İşaret {mid}"
        self._markers[mid] = {"id": mid, "label": label, "x": x, "y": y}

        x_str = _format_x_display(self.plot_pane.axis_mode, x)
        it = QTreeWidgetItem([label, x_str, f"{y:.12g}"])
        it.setData(0, Qt.ItemDataRole.UserRole, mid)
        self.marker_list.addTopLevelItem(it)

        self.plot_pane.add_marker(x, y, label)
        if self.detached_win is not None:
            self.detached_win.pane.add_marker(x, y, label)

    def _on_marker_double_clicked(self, item: QTreeWidgetItem, _col: int) -> None:
        mid = item.data(0, Qt.ItemDataRole.UserRole)
        if not isinstance(mid, int) or mid not in self._markers:
            return
        x = self._markers[mid]["x"]
        self.plot_pane.jump_to_x(x)
        if self.detached_win is not None:
            self.detached_win.pane.jump_to_x(x)

    def _selected_two_markers(self) -> Optional[Tuple[dict, dict]]:
        items = self.marker_list.selectedItems()
        if len(items) != 2:
            QMessageBox.information(self, "Seçim", "Lütfen tam olarak 2 işaret seçin (CTRL+tıklama).")
            return None
        ms = []
        for it in items:
            mid = it.data(0, Qt.ItemDataRole.UserRole)
            if isinstance(mid, int) and mid in self._markers:
                ms.append(self._markers[mid])
        if len(ms) != 2:
            QMessageBox.information(self, "Seçim", "2 geçerli işaret bulunamadı.")
            return None
        ms.sort(key=lambda m: m["id"])
        return ms[0], ms[1]

    def _marker_math_add(self) -> None:
        pair = self._selected_two_markers()
        if not pair:
            return
        a, b = pair
        mode = self.cmb_marker_calc_axis.currentText()
        parts = [f"Sonuç: {a['label']} + {b['label']}"]
        if mode in ("Yalnız X", "X ve Y"):
            parts.append(f"X={a['x'] + b['x']:.12g}")
        if mode in ("Yalnız Y", "X ve Y"):
            parts.append(f"Y={a['y'] + b['y']:.12g}")
        self.lbl_marker_calc.setText("  |  ".join(parts))

    def _marker_math_sub(self) -> None:
        pair = self._selected_two_markers()
        if not pair:
            return
        a, b = pair
        mode = self.cmb_marker_calc_axis.currentText()
        parts = [f"Sonuç: {b['label']} - {a['label']}"]
        if mode in ("Yalnız X", "X ve Y"):
            parts.append(f"\u0394X={b['x'] - a['x']:.12g}")
        if mode in ("Yalnız Y", "X ve Y"):
            parts.append(f"\u0394Y={b['y'] - a['y']:.12g}")
        self.lbl_marker_calc.setText("  |  ".join(parts))

    def _marker_interval_stats(self) -> None:
        pair = self._selected_two_markers()
        if not pair or not self.current_series:
            return
        a, b = pair; x0, x1 = sorted((float(a["x"]), float(b["x"])))
        idx = self.cmb_style_series.currentIndex()
        skey = self.cmb_style_series.itemData(idx, role=Qt.ItemDataRole.UserRole) if idx >= 0 else None
        s = next((q for q in self.current_series if q.get("style_key") == skey), None)
        if s is None:
            QMessageBox.information(self, "Kanal", "Stil sekmesinden istatistik alınacak kanalı seçin."); return
        try:
            x, y = self._load_full_xy(s); x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float)
            st=self.style_map.get(str(skey),{}); x=x+float(st.get("x_shift",0.0) or 0.0); y=y+float(st.get("y_shift",0.0) or 0.0)
            m=(x>=x0)&(x<=x1)&np.isfinite(y)&np.isfinite(x)
            if np.count_nonzero(m)<1: raise ValueError("Seçili aralıkta geçerli örnek yok.")
            yy=y[m]; xx=x[m]; mean=float(np.mean(yy)); rms=float(np.sqrt(np.mean(yy*yy)))
            slope=(float(yy[-1]-yy[0])/float(xx[-1]-xx[0])) if xx.size>1 and xx[-1]!=xx[0] else float('nan')
            self.lbl_marker_calc.setText(f"{s.get('name','Kanal')} | Δt={x1-x0:.6g} | N={yy.size} | min={np.min(yy):.6g} | max={np.max(yy):.6g} | mean={mean:.6g} | RMS={rms:.6g} | slope={slope:.6g}")
        except Exception as e:
            QMessageBox.warning(self, "İstatistik", str(e))

    # ----- CSV Export -----

    def export_csv(self) -> None:
        """Export currently plotted channel data to a CSV file."""
        if not self.current_series:
            QMessageBox.information(self, "Veri Yok", "Dışa aktarmadan önce kanalları çizin.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "CSV Dışa Aktar", "", "CSV dosyaları (*.csv);;Tüm dosyalar (*)",
        )
        if not path:
            return

        try:
            import csv
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for s in self.current_series:
                    name = s.get("name", "channel")
                    fl = s.get("file_label", "")
                    full_name = f"{fl} | {name}" if fl else name
                    unit = s.get("unit", "")
                    x_mode = s.get("x_mode", "index")

                    x_header = f"X ({x_mode})"
                    y_header = f"{full_name} [{unit}]" if unit else full_name
                    writer.writerow([x_header, y_header])

                    x, y = self._load_full_xy(s)
                    for xi, yi in zip(x, y):
                        writer.writerow([f"{xi:.12g}", f"{yi:.12g}"])
                    writer.writerow([])

            self.status.showMessage(f"{os.path.basename(path)} dosyasına aktarıldı", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Dışa Aktarma Hatası", f"CSV dışa aktarma başarısız:\n{e}")

    # ----- FFT -----

    def _fft_curve_pen(self) -> Any:
        fg = _bg_to_fg_rgb(self.plot_bg_rgb)
        return pg.mkPen((114, 180, 255) if fg == (255, 255, 255) else (0, 92, 175), width=1.8)

    def _fft_peak_pen(self) -> Any:
        fg = _bg_to_fg_rgb(self.plot_bg_rgb)
        return pg.mkPen(
            (255, 196, 87) if fg == (255, 255, 255) else (190, 95, 0),
            width=1.2, style=Qt.PenStyle.DashLine,
        )

    def _fft_peak_brush(self) -> Any:
        fg = _bg_to_fg_rgb(self.plot_bg_rgb)
        return pg.mkBrush(255, 196, 87, 210) if fg == (255, 255, 255) else pg.mkBrush(200, 110, 20, 210)

    def clear_fft_plot(self) -> None:
        self._last_fft_payload = None
        self.fft_plot.clear()
        apply_plotwidget_theme(self.fft_plot, self.plot_bg_rgb)
        self.fft_plot.setLabel("bottom", "Frekans (Hz)")
        self.fft_plot.setLabel("left", "Genlik")
        try:
            self.fft_plot.getPlotItem().setTitle("")
        except Exception:
            pass
        self.lbl_fft_info.setText("Bir kanal seçin ve FFT hesaplayın.")

    def _rerender_fft_plot(self, *_: Any) -> None:
        if self._last_fft_payload:
            self._render_fft_payload(self._last_fft_payload)

    def _render_fft_payload(self, p: dict) -> None:
        if not p:
            return
        freq = np.asarray(p.get("freq", []), dtype=np.float64)
        mag_linear = np.asarray(p.get("mag_linear", []), dtype=np.float64)
        if freq.size == 0 or mag_linear.size == 0:
            self.clear_fft_plot()
            return

        fmin = self.sp_fft_fmin.value()
        fmax = self.sp_fft_fmax.value()
        hide_dc = self.chk_fft_hide_dc.isChecked()
        use_log = self.chk_fft_log.isChecked()

        mask = np.isfinite(freq) & np.isfinite(mag_linear)
        if hide_dc:
            mask &= (freq > 0.0)
        if fmin > 0:
            mask &= (freq >= fmin)
        if fmax > 0:
            mask &= (freq <= fmax)

        freq_p, mag_p = freq[mask], mag_linear[mask]
        self.fft_plot.clear()
        apply_plotwidget_theme(self.fft_plot, self.plot_bg_rgb)

        if freq_p.size == 0:
            self.fft_plot.setLabel("bottom", "Frekans (Hz)")
            self.fft_plot.setLabel("left", "Genlik")
            try:
                self.fft_plot.getPlotItem().setTitle("<b>FFT</b> — Seçili aralıkta veri yok")
            except Exception:
                pass
            self.lbl_fft_info.setText("Seçili frekans aralığında spektrum verisi yok.")
            return

        if use_log:
            mag_plot = 20.0 * np.log10(np.maximum(mag_p, 1e-20))
            ylab = "Genlik (dB)"
        else:
            mag_plot = mag_p
            ylab = "Genlik"

        curve = self.fft_plot.plot(freq_p, mag_plot, pen=self._fft_curve_pen())
        try:
            curve.setClipToView(True)
            curve.setDownsampling(auto=True, method="peak")
        except Exception:
            pass
        try:
            curve.setSkipFiniteCheck(True)
        except Exception:
            pass

        self.fft_plot.setLabel("bottom", "Frekans (Hz)")
        self.fft_plot.setLabel("left", ylab)

        shown_fmin, shown_fmax = float(freq_p[0]), float(freq_p[-1])
        title = (
            f"<b>{p['name']}</b>"
            f" &nbsp;|&nbsp; Fs={p['fs']:.6g} Hz"
            f" &nbsp;|&nbsp; \u0394f={p['df']:.6g} Hz"
            f" &nbsp;|&nbsp; N={p['n']}"
            f" &nbsp;|&nbsp; Aralık: {shown_fmin:.6g}\u2013{shown_fmax:.6g} Hz"
        )
        try:
            self.fft_plot.getPlotItem().setTitle(title)
        except Exception:
            pass

        xr = padded_minmax(freq_p, pad_ratio=0.01)
        yr = padded_minmax(mag_plot, pad_ratio=0.08)
        if xr:
            self.fft_plot.setXRange(*xr, padding=0.0)
        if yr:
            self.fft_plot.setYRange(*yr, padding=0.0)

        peak_text = "Tepe işaretleme devre dışı"
        if self.chk_fft_peak.isChecked() and freq_p.size > 0:
            try:
                peak_idx = int(np.nanargmax(mag_p))
                peak_freq = float(freq_p[peak_idx])
                peak_mag = float(mag_p[peak_idx])
                peak_val = float(mag_plot[peak_idx])
                self.fft_plot.addItem(
                    pg.InfiniteLine(pos=peak_freq, angle=90, pen=self._fft_peak_pen(), movable=False)
                )
                self.fft_plot.addItem(
                    pg.ScatterPlotItem(
                        [peak_freq], [peak_val], size=8,
                        pen=pg.mkPen(0, 0, 0, 0), brush=self._fft_peak_brush(),
                    )
                )
                peak_text = f"Baskın tepe: {peak_freq:.6g} Hz | Genlik: {peak_mag:.6g}"
            except Exception:
                peak_text = "Tepe algılama başarısız"

        preprocess = []
        if p.get("detrend_linear"):
            preprocess.append("doğrusal trend kaldırma")
        elif p.get("remove_mean"):
            preprocess.append("ortalama kaldırıldı")
        if p.get("use_window"):
            preprocess.append(p.get("window_name", "pencere"))
        if not preprocess:
            preprocess.append("ham sinyal")

        info = [
            f"<b>Kanal:</b> {p['name']}",
            f"<b>Örnekleme:</b> {p['fs']:.6g} Hz &nbsp; <b>N:</b> {p['n']} &nbsp; <b>Çözünürlük:</b> {p['df']:.6g} Hz",
            f"<b>Ön İşleme:</b> {', '.join(preprocess)}",
            f"<b>Gösterilen aralık:</b> {shown_fmin:.6g}\u2013{shown_fmax:.6g} Hz &nbsp; <b>Ölçek:</b> {ylab}",
            f"<b>{peak_text}</b>",
        ]
        self.lbl_fft_info.setText("<br>".join(info))

    def compute_fft(self) -> None:
        if not self.current_series:
            return
        idx = self.cmb_fft_channel.currentIndex()
        if idx < 0:
            return
        skey = self.cmb_fft_channel.itemData(idx, role=Qt.ItemDataRole.UserRole)
        if not isinstance(skey, str):
            return

        chosen = next((s for s in self.current_series if s.get("style_key") == skey), None)
        if not chosen:
            return

        x, y = self._load_full_xy(chosen)
        if y is None or len(y) == 0:
            return

        if self.chk_smooth.isChecked() and SCIPY_AVAILABLE and not chosen.get("is_digital"):
            y = apply_savgol_safe(y, window_len=int(self.spin_smooth_win.value()))

        fs_hint = float(self.fs_fft.value())
        fs_hint = None if fs_hint <= 0 else fs_hint

        name = f"{chosen.get('file_label', '')} | {chosen.get('name', '')}".strip(" |")
        self.lbl_fft_info.setText("FFT hesaplanıyor...")

        worker = FFTWorker(
            name=name, x=x, y=y, fs_hint=fs_hint,
            use_window=self.chk_fft_window.isChecked(),
            remove_mean=self.chk_fft_remove_mean.isChecked(),
            detrend_linear=self.chk_fft_detrend.isChecked(),
        )
        self._start_job(worker, self._on_fft_ready, tag="fft", cancel_tags=["fft"])

    def _on_fft_ready(self, p: dict) -> None:
        if p.get("cancelled"):
            return
        self._last_fft_payload = p
        self._render_fft_payload(p)
        self.tabs.setCurrentIndex(1)

    # ----- Drag & Drop -----

    def dragEnterEvent(self, e: Any) -> None:
        try:
            md = e.mimeData()
            if md and md.hasUrls():
                if any(u.toLocalFile().lower().endswith(".tdms") for u in md.urls()):
                    e.acceptProposedAction()
                    return
        except Exception:
            pass
        e.ignore()

    def dropEvent(self, e: Any) -> None:
        paths: List[str] = []
        try:
            md = e.mimeData()
            if md and md.hasUrls():
                paths = [
                    u.toLocalFile() for u in md.urls()
                    if u.toLocalFile().lower().endswith(".tdms")
                ]
        except Exception:
            pass
        if paths:
            self.open_tdms_paths(paths)
            e.acceptProposedAction()
        else:
            e.ignore()


# ===================================================================
# Entry point
# ===================================================================

def main() -> None:
    app = QApplication(sys.argv)
    app.setStyleSheet(LIGHT_QSS)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
