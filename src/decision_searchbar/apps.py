"""The apps GNOME's overview search can launch: every `Gio.AppInfo` that `should_show()`."""

from __future__ import annotations

from dataclasses import dataclass

import gi

gi.require_version("Gio", "2.0")
from gi.repository import Gio


@dataclass(frozen=True)
class App:
    id: str
    name: str
    generic: str
    description: str
    keywords: tuple[str, ...]

    def describe(self, style: str = "full") -> str:
        """The option text the model reads for this app (the name is the option key)."""
        if style == "name":
            return ""
        parts = [p for p in (self.generic, self.description) if p]
        if style == "full" and self.keywords:
            parts.append(", ".join(self.keywords))
        return ". ".join(parts)


def installed_apps() -> list[App]:
    apps = []
    for info in Gio.AppInfo.get_all():
        if not info.should_show():
            continue
        apps.append(App(id=info.get_id(), name=info.get_display_name(), generic=info.get_generic_name() or "",
                        description=info.get_description() or "", keywords=tuple(info.get_keywords() or ())))
    return sorted(apps, key=lambda a: a.name.lower())


def launch(app_id: str) -> None:
    Gio.DesktopAppInfo.new(app_id).launch([], None)
