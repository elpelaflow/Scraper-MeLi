"""Simple Tkinter interface to trigger Mercado Libre scraping runs."""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk

from config_utils import MAX_PAGES_DEFAULT, load_search_query, resolve_max_pages, save_search_query


BASE_DIR = Path(__file__).resolve().parent


class ScraperUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()

        self.title("Scraper Mercado Libre")
        self.geometry("500x320")
        self.resizable(False, False)

        self.query_var = tk.StringVar(value=load_search_query())
        self.status_var = tk.StringVar(value="Ingrese un término de búsqueda.")
        self.max_pages_var = tk.IntVar(value=MAX_PAGES_DEFAULT)
        self.plan_var = tk.StringVar()

        self._build_widgets()

    def _build_widgets(self) -> None:
        padding = {"padx": 12, "pady": 6}

        title = ttk.Label(self, text="Generador de búsquedas", font=("Arial", 14, "bold"))
        title.pack(**padding)

        description = ttk.Label(
            self,
            text="Escriba la búsqueda para Mercado Libre Argentina y presione",
        )
        description.pack(padx=12, pady=(0, 4))

        description2 = ttk.Label(self, text="\"Generar búsquedas\" para ejecutar el scraping.")
        description2.pack(padx=12, pady=(0, 12))

        entry_frame = ttk.Frame(self)
        entry_frame.pack(fill=tk.X, padx=12)

        entry_label = ttk.Label(entry_frame, text="Búsqueda:")
        entry_label.pack(side=tk.LEFT)

        entry = ttk.Entry(entry_frame, textvariable=self.query_var)
        entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(8, 0))
        entry.focus()

        max_pages_frame = ttk.Frame(self)
        max_pages_frame.pack(fill=tk.X, padx=12, pady=(0, 8))

        max_pages_label = ttk.Label(max_pages_frame, text="Máx. páginas a recorrer:")
        max_pages_label.pack(side=tk.LEFT)

        spinbox = ttk.Spinbox(
            max_pages_frame,
            from_=1,
            to=50,
            textvariable=self.max_pages_var,
            width=5,
            command=self._update_plan_label,
        )
        spinbox.pack(side=tk.LEFT, padx=(8, 0))

        self.max_pages_var.trace_add("write", self._update_plan_label)
        self._update_plan_label()

        plan_label = ttk.Label(self, textvariable=self.plan_var)
        plan_label.pack(padx=12, pady=(0, 6))

        buttons_frame = ttk.Frame(self)
        buttons_frame.pack(pady=8)

        self.generate_button = ttk.Button(
            buttons_frame,
            text="Generar búsquedas",
            command=self._on_generate_clicked,
        )
        self.generate_button.pack(side=tk.LEFT, padx=(0, 6))

        self.dashboard_button = ttk.Button(
            buttons_frame,
            text="Ver dashboard",
            command=self._on_dashboard_clicked,
        )
        self.dashboard_button.pack(side=tk.LEFT)

        status_label = ttk.Label(self, textvariable=self.status_var)
        status_label.pack(**padding)

        self.output_text = tk.Text(self, height=10, wrap=tk.WORD, state=tk.DISABLED)
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))


    def _on_generate_clicked(self) -> None:
        search_query = self.query_var.get().strip()

        if not search_query:
            self.status_var.set("Por favor, ingrese un término de búsqueda.")
            return

        save_search_query(search_query)
        
        self.status_var.set("Ejecutando scraping...")
        self.generate_button.config(state=tk.DISABLED)
        self._write_output("")

        threading.Thread(
            target=self._run_crawler,
            args=(search_query,),
            daemon=True,
        ).start()

    def _run_crawler(self, search_query: str) -> None:
        max_pages = self._get_planned_max_pages()
        command = [
            sys.executable,
            "crawl.py",
            "--query",
            search_query,
            "--max-pages",
            str(max_pages),
        ]

        try:
            process = subprocess.run(
                command,
                cwd=BASE_DIR,
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception as exc:  # pylint: disable=broad-except
            self.after(0, self._handle_failure, f"Error al ejecutar el crawler: {exc}")
            return

        combined_output = (process.stdout or "") + (process.stderr or "")
        self.after(0, self._finalize_run, process.returncode, combined_output)

    def _finalize_run(self, returncode: int, output: str) -> None:
        if returncode == 0:
            self.status_var.set("Scraping finalizado con éxito.")
        else:
            self.status_var.set("Scraping finalizado con errores. Revise el detalle.")

        self._write_output(output)
        self.generate_button.config(state=tk.NORMAL)

    def _handle_failure(self, message: str) -> None:
        self.status_var.set(message)
        self.generate_button.config(state=tk.NORMAL)

    def _on_dashboard_clicked(self) -> None:
        self.status_var.set("Abriendo dashboard...")
        self.dashboard_button.config(state=tk.DISABLED)

        threading.Thread(target=self._run_dashboard, daemon=True).start()

    def _run_dashboard(self) -> None:
        command = ["streamlit", "run", "dashboard/dashboard.py"]

        try:
            subprocess.Popen(command, cwd=BASE_DIR)
        except Exception as exc:  # pylint: disable=broad-except
            self.after(0, self._handle_dashboard_failure, exc)
        else:
            self.after(0, self._handle_dashboard_success)

    def _handle_dashboard_success(self) -> None:
        self.status_var.set("Dashboard iniciado. Revise la consola para más detalles.")
        self.dashboard_button.config(state=tk.NORMAL)

    def _handle_dashboard_failure(self, exc: Exception) -> None:  # pylint: disable=broad-except
        self.status_var.set(f"Error al abrir el dashboard: {exc}")
        self.dashboard_button.config(state=tk.NORMAL)

    def _write_output(self, message: str) -> None:
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        if message:
            self.output_text.insert(tk.END, message)
        self.output_text.config(state=tk.DISABLED)

    def _update_plan_label(self, *_: object) -> None:
        try:
            current = self.max_pages_var.get()
        except tk.TclError:
            current = MAX_PAGES_DEFAULT
        planned = resolve_max_pages(ui_value=current)
        if current != planned:
            self.max_pages_var.set(planned)
            return
        suffix = "página" if planned == 1 else "páginas"
        self.plan_var.set(f"Planeado: {planned} {suffix}")

    def _get_planned_max_pages(self) -> int:
        try:
            current = self.max_pages_var.get()
        except tk.TclError:
            current = MAX_PAGES_DEFAULT
        planned = resolve_max_pages(ui_value=current)
        if planned != current:
            self.max_pages_var.set(planned)
        suffix = "página" if planned == 1 else "páginas"
        self.plan_var.set(f"Planeado: {planned} {suffix}")
        return planned


if __name__ == "__main__":
    app = ScraperUI()
    app.mainloop()