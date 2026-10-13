"""A standalone launcher window: type a request, results rank after a 300 ms pause, Enter launches."""

from __future__ import annotations

import threading

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gio, GLib, Gtk

from .apps import installed_apps, launch

SHOWN = 8


class Launcher(Adw.ApplicationWindow):
    def __init__(self, app: Adw.Application):
        super().__init__(application=app, title="Decision search", default_width=560, default_height=480)
        self.model, self.apps, self.ranked, self.seq = None, installed_apps(), [], 0
        self.lock = threading.Lock()  # one forward pass at a time on the GPU

        self.entry = Gtk.SearchEntry(placeholder_text="Loading model…", search_delay=300, hexpand=True)
        self.entry.connect("search-changed", self.on_search)
        self.entry.connect("activate", lambda *_: self.open(0))
        self.entry.connect("stop-search", lambda *_: self.close())
        self.status = Gtk.Label(xalign=0, css_classes=["dim-label", "caption"])
        self.list = Gtk.ListBox(css_classes=["boxed-list"], selection_mode=Gtk.SelectionMode.NONE)
        self.list.connect("row-activated", lambda _, row: self.open(row.get_index()))

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12, margin_top=12, margin_bottom=12,
                      margin_start=12, margin_end=12)
        for child in (self.entry, self.status, Gtk.ScrolledWindow(child=self.list, vexpand=True)):
            box.append(child)
        self.set_content(box)
        self.entry.grab_focus()
        try:
            from . import open_model

            model = open_model()  # spawned here: a llama-server dies with the thread that started it
        except Exception as e:  # noqa: BLE001 - shown in the window
            self.status.set_text(f"Could not start the model: {e}")
            return
        threading.Thread(target=self.load, args=(model,), daemon=True).start()

    def load(self, model) -> None:
        from .rank import rank

        try:
            model.ready()
            rank(model, "warm up", self.apps)  # the first GPU call is several times slower
        except Exception as e:  # noqa: BLE001 - shown in the window instead of a silent dead thread
            GLib.idle_add(self.status.set_text, f"Could not load the model: {e}")
            return
        GLib.idle_add(self.loaded, model)

    def loaded(self, model) -> None:
        self.model = model
        self.entry.set_placeholder_text("What do you want to do?")
        self.status.set_text(f"{len(self.apps)} apps · {model.device}")
        if self.entry.get_text().strip():
            self.on_search(self.entry)

    def on_search(self, entry: Gtk.SearchEntry) -> None:
        query = entry.get_text().strip()
        self.seq += 1
        if not query or self.model is None:
            self.show([], "")
            return
        self.status.set_text("Ranking…")
        threading.Thread(target=self.rank, args=(self.seq, query), daemon=True).start()

    def rank(self, seq: int, query: str) -> None:
        import time

        from .rank import rank

        with self.lock:
            if seq != self.seq:  # a newer query arrived while this one waited
                return
            start = time.perf_counter()
            ranked = rank(self.model, query, self.apps)
            ms = (time.perf_counter() - start) * 1000
        GLib.idle_add(lambda: seq == self.seq and self.show(ranked, f"{ms:.0f} ms"))

    def show(self, ranked: list, status: str) -> None:
        self.ranked = ranked[:SHOWN]
        self.list.remove_all()
        for app, score in self.ranked:
            info = Gio.DesktopAppInfo.new(app.id)
            row = Adw.ActionRow(title=GLib.markup_escape_text(app.name),
                                subtitle=GLib.markup_escape_text(app.description or app.generic), activatable=True)
            if info and info.get_icon():
                row.add_prefix(Gtk.Image(gicon=info.get_icon(), pixel_size=32))
            row.add_suffix(Gtk.Label(label=f"{score:.2f}",
                                     css_classes=["dim-label", "numeric"]))
            self.list.append(row)
        if status:
            self.status.set_text(status)

    def open(self, index: int) -> None:
        if index < len(self.ranked):
            launch(self.ranked[index][0].id)
            self.close()


def run() -> None:
    app = Adw.Application(application_id="io.github.carlulsoe.DecisionSearch")
    app.connect("activate", lambda a: Launcher(a).present())
    app.run([])
