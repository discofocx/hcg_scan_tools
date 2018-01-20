""" Linear sequence trans-coder,
    used mainly for generating animated GIFs from a camera rig """

# imports
import os
import sys
import time

from PyQt4 import QtGui, QtCore
import qdarkstyle

from transcode import transcode_ops as tr_ops


__author__ = '__discofocx__'
__copyright__ = 'Copyright 2018, HCG Technologies'
__version__ = '0.1.2'
__email__ = 'gsorchin@gmail.com'
__status__ = 'alpha'

# Globals
gDEBUG = False
gWIDTH = 512

if gDEBUG:
    icon = 'hcg.png'
else:
    pass  # icon = os.path.join(sys._MEIPASS,'hcg.png') TODO Set icon remotely


# --- Classes --- #


class AppWindow(QtGui.QDialog):

    # Class Attributes
    water_mark = 'watermarks\\hcg_tech.png'

    def __init__(self):
        super(AppWindow, self).__init__()
        self.setFixedWidth(gWIDTH)
        self.setWindowTitle('gTools 0.1.2')
        # self.setWindowIcon(QtGui.QIcon(icon))  # TODO Fix icon placement

        # Instance Attributes
        self.valid_sequence = False
        self.process_thread = None
        self.seq_contents = None
        self.last_known_path = os.getcwd()

        # Build GUI
        self.home()

    def __call__(self):
        self.show()

    def home(self):
        # Main layout
        self.setLayout(QtGui.QVBoxLayout())
        self.layout().setContentsMargins(4, 4, 4, 4)
        self.layout().setSpacing(4)
        self.layout().setAlignment(QtCore.Qt.AlignTop)

        # Load GIF sequence frame
        gif_frame = QtGui.QFrame()
        gif_frame.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        gif_frame.setLayout(QtGui.QHBoxLayout())
        gif_frame.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        gif_frame.layout().setContentsMargins(4, 4, 4, 4)
        gif_frame.layout().setSpacing(8)

        fld_path_lbl = QtGui.QLabel(' Path:')
        self.fld_line = QtGui.QLineEdit()
        self.fld_line.setText('Browse for a folder that contains a sequence ...')
        self.fld_line.setEnabled(False)
        fld_btn = QtGui.QPushButton('Browse')

        # Add Load GIF sequence layout widgets
        gif_frame.layout().addWidget(fld_path_lbl)
        gif_frame.layout().addWidget(self.fld_line)
        gif_frame.layout().addWidget(fld_btn)

        # Options GIF sequence frame
        opts_frame = QtGui.QFrame()
        opts_frame.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        opts_frame.setLayout(QtGui.QVBoxLayout())
        opts_frame.layout().setContentsMargins(4, 4, 4, 4)
        opts_frame.layout().setSpacing(4)
        opts_frame.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)

        # Top options layout
        top_opts = QtGui.QHBoxLayout()
        top_opts.layout().setContentsMargins(0, 0, 0, 0)
        top_opts.layout().setSpacing(8)
        top_opts.layout().setAlignment(QtCore.Qt.AlignCenter)
        # top_opts.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)

        # Bottom options layout
        bottom_opts = QtGui.QHBoxLayout()
        bottom_opts.layout().setContentsMargins(4, 4, 4, 4)
        bottom_opts.layout().setSpacing(8)
        # bottom_opts.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)

        # Size
        self.size_combo = QtGui.QComboBox()
        self.size_combo.setFixedHeight(28)
        self.size_combo.addItem('1/1 SZ')
        self.size_combo.addItem('1/2 SZ')
        self.size_combo.addItem('1/4 SZ')
        self.size_combo.addItem('1/8 SZ')

        # Duration
        dur_lbl = QtGui.QLabel('Duration (sec):')
        self.dur_line = QtGui.QLineEdit()
        self.dur_line.setFixedWidth(32)
        self.dur_line.setFixedHeight(28)
        self.dur_line.setMaxLength(2)
        self.dur_line.setAlignment(QtCore.Qt.AlignCenter)

        # Duration regex
        reg_ex = QtCore.QRegExp("^[0-9]+$")
        number_validator = QtGui.QRegExpValidator(reg_ex, self.dur_line)
        self.dur_line.setValidator(number_validator)

        # Behavior mode
        self.mode_combo = QtGui.QComboBox()
        self.mode_combo.setFixedHeight(28)
        self.mode_combo.addItem('Loop')
        self.mode_combo.addItem('Boom')

        # Crop
        crop_lbl = QtGui.QLabel('Crop:')
        self.crop_check = QtGui.QCheckBox()

        # Watermark
        w_mark_lbl = QtGui.QLabel('Watermark:')
        self.w_mark_check = QtGui.QCheckBox()
        self.w_mark_check.setEnabled(False)

        # Rename
        rename_lbl = QtGui.QLabel('Rename:')
        rename_check = QtGui.QCheckBox()
        rename_check.setEnabled(False)
        self.rename_line = SensitiveLineEdit('Type new name ...')

        # Add widgets to top options frame
        top_opts.layout().addWidget(self.size_combo)
        top_opts.layout().addWidget(self.mode_combo)
        top_opts.layout().addWidget(dur_lbl)
        top_opts.layout().addWidget(self.dur_line)
        top_opts.layout().addWidget(crop_lbl)
        top_opts.layout().addWidget(self.crop_check)
        top_opts.layout().addWidget(w_mark_lbl)
        top_opts.layout().addWidget(self.w_mark_check)

        # Add widgets to bottom options frame
        bottom_opts.layout().addWidget(rename_lbl)
        # bottom_opts.layout().addWidget(rename_check)
        bottom_opts.layout().addWidget(self.rename_line)

        opts_frame.layout().addLayout(top_opts)
        opts_frame.layout().addLayout(bottom_opts)

        # Progress bar frame
        prg_frame = QtGui.QFrame()
        prg_frame.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        prg_frame.setLayout(QtGui.QHBoxLayout())
        prg_frame.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        prg_frame.layout().setContentsMargins(4, 4, 4, 4)
        prg_frame.layout().setSpacing(4)

        self.prg_bar = QtGui.QProgressBar()
        self.run_btn = QtGui.QPushButton('Run')
        self.run_btn.setEnabled(False)

        # Add Progress bar widgets
        prg_frame.layout().addWidget(self.prg_bar)
        prg_frame.layout().addWidget(self.run_btn)

        # Logger frame
        logger_frame = QtGui.QFrame()
        logger_frame.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        logger_frame.setLayout(QtGui.QHBoxLayout())
        logger_frame.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        logger_frame.layout().setContentsMargins(4, 4, 4, 4)
        logger_frame.layout().setSpacing(4)

        self.logger_text = Logger()
        logger_frame.layout().addWidget(self.logger_text)

        # Footer layout
        footer = QtGui.QFrame()
        footer.setLayout(QtGui.QHBoxLayout())
        footer.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        footer.layout().setAlignment(QtCore.Qt.AlignRight)

        footer_lbl = QtGui.QLabel('\u00a9 2017 HCG Technologies. __discofocx__')

        footer.layout().addWidget(footer_lbl)

        # Add Main layout widgets
        self.layout().addWidget(gif_frame)
        self.layout().addWidget(opts_frame)
        self.layout().addWidget(prg_frame)
        self.layout().addWidget(logger_frame)
        self.layout().addWidget(footer)

        # Widget Behaviours
        fld_btn.clicked.connect(self.browse_for_sequence)
        self.dur_line.textChanged.connect(self.check_requirements)
        self.rename_line.textChanged.connect(self.check_requirements)
        self.run_btn.clicked.connect(self.pre_process_sequence)

    def browse_for_sequence(self):

        seq_path = QtGui.QFileDialog.getExistingDirectory(self,
                                                          caption='Select folder',
                                                          directory=self.last_known_path)

        try:
            self.seq_contents = tr_ops.build_data_list(seq_path)
        except ValueError as e:
            self.write_to_logger(e)
            self.valid_sequence = False
            self.check_requirements()
        except NotADirectoryError as e:
            self.write_to_logger(e)
            self.fld_line.setText('Browse for a folder that contains a sequence ...')
            self.valid_sequence = False
            self.check_requirements()
        else:
            self.last_known_path = seq_path
            seq_path = format_full_path(seq_path)
            self.fld_line.setText(seq_path)
            self.write_to_logger('Found {0} images to process'.format(len(self.seq_contents)))
            self.run_btn.setEnabled(True)
            self.valid_sequence = True
            self.check_requirements()

    def check_requirements(self):

        if self.valid_sequence and self.dur_line.hasAcceptableInput() and self.rename_line.hasAcceptableInput():
            self.run_btn.setEnabled(True)
        else:
            self.run_btn.setEnabled(False)

    def write_to_logger(self, message):

        msg = '<<< {0}'.format(message)
        self.logger_text.append(msg)

    def pre_process_sequence(self):

        sequence_manifest = {'data': self.seq_contents,
                             'size': str(self.size_combo.currentText()),
                             'mode': str(self.mode_combo.currentText()),
                             'duration': int(self.dur_line.text()),
                             'crop': self.crop_check.isChecked(),
                             'watermark': (self.w_mark_check.isChecked(), self.water_mark),
                             'rename': self.rename_line.text()}

        if gDEBUG:
            for k, v in sequence_manifest.items():
                print(k, v)

        if sequence_manifest['mode'] == 'Boom':
            self.prg_bar.setMaximum((len(sequence_manifest['data']) * 2) - 1)

        else:
            self.prg_bar.setMaximum((len(sequence_manifest['data'])))

        # Process call to the trans-code ops library
        self.process_thread = ProcessSequence(**sequence_manifest)
        self.connect(self.process_thread, QtCore.SIGNAL('finished_frame(QString)'), self.finished_frame)
        self.connect(self.process_thread, QtCore.SIGNAL('finished()'), self.finished_sequence)
        self.process_thread.start()

        # process_start = time.time()

        # process_end = time.time()

        # self.prg_bar.setValue(0)
        # self.write_to_logger('Finished encoding in {0} seconds'.format(round(process_end - process_start, 2)))
        # self.write_to_logger('Ready to process again.')

    def finished_frame(self, string):
        self.prg_bar.setValue(self.prg_bar.value() + 1)

    def finished_sequence(self):
        self.prg_bar.setValue(0)
        self.write_to_logger('Thread finished')


class ProcessSequence(QtCore.QThread):
    def __init__(self, **manifest):
        super(ProcessSequence, self).__init__()

        self.manifest = dict()
        for k, v in manifest.items():
            self.manifest[k] = v

    def __del__(self):
        self.wait()

    def run(self):
        process = tr_ops.process_sequence(**self.manifest)

        for i, t in process:
            p = ((i + 1) * 100) / t
            self.emit(QtCore.SIGNAL('finished_frame(QString)'), str(i))


class Logger(QtGui.QTextEdit):
    def __init__(self):
        super(Logger, self).__init__()

        self.setStyleSheet('background-color: rgb(30, 33, 36); color: rgb(61, 174, 233)')
        #self.setFixedHeight(64)
        self.setReadOnly(True)
        self.setText('<<< Application Started')


class SensitiveLineEdit(QtGui.QLineEdit):
    def __init__(self, idle_text):
        super(SensitiveLineEdit, self).__init__()

        self.setEnabled(False)
        self.idle_text = idle_text
        self.setMouseTracking(True)
        self.setText(self.idle_text)

        reg_ex = QtCore.QRegExp("^(?!^_)[a-zA-Z_0-9]+")
        text_validator = QtGui.QRegExpValidator(reg_ex, self)
        self.setValidator(text_validator)

    def enterEvent(self, event):
        if self.text() != self.idle_text:
            pass
        else:
            self.clear()
            self.setEnabled(True)

    def leaveEvent(self, event):
        if self.text():
            pass
        else:
            self.setText(self.idle_text)
            self.setEnabled(False)


class HSplitter(QtGui.QWidget):
    def __init__(self):
        super(HSplitter, self).__init__()

        self.setMinimumHeight(2)
        self.setLayout(QtGui.QHBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)
        self.layout().setAlignment(QtCore.Qt.AlignHCenter)

        line = QtGui.QFrame()
        line.setFrameStyle(QtGui.QFrame.VLine)
        line.setFixedWidth(1)

        self.layout().addWidget(line)


# --- Helper functions --- #


def format_full_path(l_string):
    """ Helper function
        Returns a short version of a full system path, for display only """

    s_chunks = l_string.split('\\')
    head, tail = (s_chunks[:2], s_chunks[-2:])
    head = '\\'.join(head)
    tail = '\\'.join(tail)
    f_string = head + '\\ ... \\' + tail

    return f_string


# --- Main functions --- #


def run():
    """ Main execution """
    app = QtGui.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet(pyside=False))
    GUI = AppWindow()
    GUI()

    #app.exec_()

    sys.exit(app.exec_())


# --- Entry Point --- #
if __name__ == '__main__':
    run()
