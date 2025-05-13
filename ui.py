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
        self.image_label.image = self._photo  # previene garbage collection

    def update_tracks(self, tracks):
        """Actualiza la lista de tracks activos en formato resumen completo."""
        self.listbox.delete(0, tk.END)
        personas = {}
        for det in tracks:
            tid = det['track_id']
            label = det['label']
            conf = det['conf']
            if tid not in personas:
                personas[tid] = {}
            personas[tid][label] = conf

        self.listbox.insert(tk.END, f"Total detectadas: {len(personas)} persona(s)")
        for tid, items in personas.items():
            self.listbox.insert(tk.END, f"\nPersona {tid}:")
            for base in ["helmet", "goggles", "vest"]:
                pos = items.get(base)
                neg = items.get(f"no-{base}")
                if neg is not None:
                    self.listbox.insert(tk.END, f"  no-{base}: {neg:.2f}")
                elif pos is not None:
                    self.listbox.insert(tk.END, f"  {base}: {pos:.2f}")
                else:
                    self.listbox.insert(tk.END, f"  {base}: no detectado")

    def run(self):
        """Arranca el bucle principal de Tkinter."""
        self.root.mainloop()
