import sys
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QVBoxLayout, QHBoxLayout, QGridLayout,
                               QLabel, QPushButton, QTableWidget, QTableWidgetItem,
                               QGroupBox, QSplitter, QFileDialog, QLineEdit,
                               QInputDialog, QMessageBox)
from pathlib import Path


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("传送带检测界面")
        self.resize(1000, 650)

        # ---- 中心 widget ----
        central = QWidget()
        self.setCentralWidget(central)
        master = QHBoxLayout(central)

        # 左侧：图像显示
        left_box = QGroupBox("图像显示")
        left_box.setMinimumWidth(400)
        left_lay = QVBoxLayout(left_box)

        self.lbl_origin = QLabel("原图")
        self.lbl_origin.setMinimumHeight(250)
        self.lbl_origin.setAlignment(Qt.AlignCenter)
        self.lbl_origin.setStyleSheet("border:1px solid #aaa;background:#f8f8f8;")

        self.lbl_binary = QLabel("二值化")
        self.lbl_binary.setMinimumHeight(250)
        self.lbl_binary.setAlignment(Qt.AlignCenter)
        self.lbl_binary.setStyleSheet("border:1px solid #aaa;background:#f8f8f8;")

        left_lay.addWidget(self.lbl_origin)
        left_lay.addWidget(self.lbl_binary)

        # 右上：导入/输出区域
        top_bar = QWidget()
        top_v = QVBoxLayout(top_bar)

        # 导入背景模板
        self.btn_import_bg = QPushButton("导入背景模板")
        self.line_bg_path = QLineEdit()
        self.line_bg_path.setPlaceholderText("（文件路径）")
        self.line_bg_path.setReadOnly(True)
        top_v.addWidget(self.btn_import_bg)
        top_v.addWidget(self.line_bg_path)

        # 输出文件夹
        self.btn_out_dir = QPushButton("输出文件夹")
        self.line_out_path = QLineEdit()
        self.line_out_path.setPlaceholderText("（文件路径）")
        self.line_out_path.setReadOnly(True)
        top_v.addWidget(self.btn_out_dir)
        top_v.addWidget(self.line_out_path)

        # 右侧：计数表 + 增删按钮 + 控制按钮
        right_box = QGroupBox("计数表（实时更新）")
        right_master = QVBoxLayout(right_box)

        # 表格
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["种类", "数量"])
        # for txt in ["种类1", "种类2", "种类3"]:
        #     self.add_kind_row(txt, 0)
        right_master.addWidget(self.table)

        # 增删按钮
        kind_btn_bar = QWidget()
        kind_btn_lay = QHBoxLayout(kind_btn_bar)
        self.btn_add_kind = QPushButton("新增种类")
        self.btn_del_kind = QPushButton("删除选中种类")
        kind_btn_lay.addWidget(self.btn_add_kind)
        kind_btn_lay.addWidget(self.btn_del_kind)
        right_master.addWidget(kind_btn_bar)

        # 2×2 控制按钮
        grid = QGridLayout()
        self.btn_start = QPushButton("启动传送带")
        self.btn_pause = QPushButton("暂停")
        self.btn_stop = QPushButton("急停")
        self.btn_vib = QPushButton("振动盘启动")
        self.btn_stop.setStyleSheet("background:#e74c3c;color:white;")
        self.btn_pause.setStyleSheet("background:#42ceff;color:white;")

        grid.addWidget(self.btn_start, 0, 0)
        grid.addWidget(self.btn_pause, 0, 1)
        grid.addWidget(self.btn_vib, 1, 0)
        grid.addWidget(self.btn_stop, 1, 1)
        right_master.addLayout(grid)

        # 组合右侧整体
        right_wrap = QWidget()
        right_wrap.setLayout(right_master)
        right_out = QVBoxLayout()
        right_out.addWidget(top_bar)
        right_out.addWidget(right_wrap)

        # splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_box)
        right_container = QWidget()
        right_container.setLayout(right_out)
        splitter.addWidget(right_container)
        splitter.setSizes([500, 400])
        master.addWidget(splitter)

        # ---- 信号 ----
        self.btn_import_bg.clicked.connect(self.slot_import_bg)
        self.btn_out_dir.clicked.connect(self.slot_out_dir)
        self.btn_add_kind.clicked.connect(self.slot_add_kind)
        self.btn_del_kind.clicked.connect(self.slot_del_kind)
        self.btn_start.clicked.connect(lambda: print("启动传送带"))
        self.btn_pause.clicked.connect(lambda: print("暂停"))
        self.btn_stop.clicked.connect(lambda: print("急停"))
        self.btn_vib.clicked.connect(lambda: print("振动盘启动"))

    # ------------ 工具函数 ------------
    def add_kind_row(self, name: str, count: int):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(name))
        self.table.setItem(row, 1, QTableWidgetItem(str(count)))

    # ------------ 槽函数 ------------
    def slot_import_bg(self):
        folder = QFileDialog.getExistingDirectory(self, "选择背景模板文件夹")
        if folder:
            self.line_bg_path.setText(folder)
            suffix = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
            bg_list = [str(p) for p in Path(folder).iterdir()
                       if p.is_file() and p.suffix.lower() in suffix]
            bg_list.sort()
            print(f"共加载 {len(bg_list)} 张背景模板")
            for p in bg_list:
                print("  ", p)

    def slot_out_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if folder:
            self.line_out_path.setText(folder)
            print("输出目录：", folder)

    def slot_add_kind(self):
        name, ok = QInputDialog.getText(self, "新增种类", "请输入种类名称：")
        if ok and name.strip():
            self.add_kind_row(name.strip(), 0)

    def slot_del_kind(self):
        rows = set(item.row() for item in self.table.selectedItems())
        if not rows:
            QMessageBox.information(self, "提示", "请先选中要删除的行！")
            return
        for r in sorted(rows, reverse=True):
            self.table.removeRow(r)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())