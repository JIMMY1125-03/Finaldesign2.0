from PyQt5.QtWidgets import *
import A_windows
import A_login
import A_register
import A_system
import sys

if __name__ == '__main__':
    app = QApplication(sys.argv)
    login = A_login.LoginUi()
    register = A_register.RegisterUi()
    system = A_system.SystemUi()
    A_windows.Windows.login = login
    A_windows.Windows.register = register
    A_windows.Windows.system = system
    A_windows.Windows.login.show()
    sys.exit(app.exec_())
