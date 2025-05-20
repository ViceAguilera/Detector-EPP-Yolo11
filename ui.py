"""
    Módulo de interfaz gráfica para la aplicación de detección de EPP.
"""
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2

class AppUI:
    """
    Interfaz gráfica local para mostrar el video procesado y el estado de los tracks.
    """
    def __init__(self, width=800, height=600):
        self.root = tk.Tk()
        self.root.title("Control de Acceso - EPP Tracker")

        # Canvas para mostrar el video
        self.image_label = tk.Label(self.root)
        self.image_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Panel lateral para lista de tracks y controles
        sidebar = tk.Frame(self.root)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Label(sidebar, text="Resumen de Personas", font=(None, 12, 'bold')).pack(pady=(10,0))
        self.listbox = tk.Listbox(sidebar, width=40, font=("Courier New", 9))
        self.listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Botón para pausar/reanudar
        self.btn_pause = tk.Button(sidebar, text="Pausar", command=self.toggle_pause)
        self.btn_pause.pack(fill=tk.X, padx=5, pady=(0,10))

        self.paused = False
        self.photo = None
        self._img_id = None

    def toggle_pause(self):
        """Alterna el estado de pausa."""
        self.paused = not self.paused
        self.btn_pause.config(text="Reanudar" if self.paused else "Pausar")

    def update_frame(self, frame):
        if self.paused:
            return

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        self._photo = ImageTk.PhotoImage(image=img)
        self.image_label.configure(image=self._photo)
        self.image_label.image = self._photo

    def update_tracks(self, resumen: str):
        self.listbox.delete(0, tk.END)
        for line in resumen.split('\n'):
            self.listbox.insert(tk.END, line)

    def run(self):
        """Arranca el bucle principal de Tkinter."""
        self.root.mainloop()
