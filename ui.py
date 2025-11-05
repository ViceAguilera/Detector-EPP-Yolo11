"""
    Módulo de interfaz gráfica para la aplicación de detección de EPP.
"""
import threading
import tkinter as tk
from typing import Optional

from PIL import Image, ImageTk
import cv2


class AppUI:
    """
    Interfaz gráfica local para mostrar el video procesado y el estado de los tracks.
    """
    def __init__(self, width: int = 800, height: int = 600, stop_event: Optional[threading.Event] = None):
        self._stop_event = stop_event

        self.root = tk.Tk()
        self.root.title("Control de Acceso - EPP Tracker")
        self.root.geometry(f"{width}x{height}")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Canvas para mostrar el video
        self.image_label = tk.Label(self.root)
        self.image_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Panel lateral para lista de tracks y controles
        sidebar = tk.Frame(self.root)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Label(sidebar, text="Resumen de Personas", font=(None, 12, 'bold')).pack(pady=(10, 0))
        self.listbox = tk.Listbox(sidebar, width=40, font=("Courier New", 9))
        self.listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self._photo = None

    def _on_close(self):
        if self._stop_event is not None:
            self._stop_event.set()
        # Cerrar ventana de forma segura
        self.root.quit()
        self.root.destroy()

    def request_close(self):
        """Solicita el cierre de la ventana desde otro hilo."""
        self.root.after(0, self._on_close)

    def update_frame(self, frame):
        if not self.root.winfo_exists():
            return
        try:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            self._photo = ImageTk.PhotoImage(image=img)
            self.image_label.configure(image=self._photo)
            self.image_label.image = self._photo
        except tk.TclError:
            pass  # La ventana se cerró durante la actualización

    def update_tracks(self, resumen: str):
        if not self.root.winfo_exists():
            return
        try:
            self.listbox.delete(0, tk.END)
            for line in resumen.split('\n'):
                self.listbox.insert(tk.END, line)
        except tk.TclError:
            pass

    def run(self):
        """Arranca el bucle principal de Tkinter."""
        self.root.mainloop()
