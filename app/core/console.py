from __future__ import annotations

from rich.console import Console
from rich.table import Table

from app.core.state import Readiness


def render_startup_status(readiness: Readiness) -> None:
    table = Table(title="Backend startup", show_header=True, header_style="bold cyan")
    table.add_column("Service")
    table.add_column("Status")
    table.add_column("Details")

    for name, item in readiness.items().items():
        status = "[green]ready[/green]" if item.ready else "[yellow]disabled[/yellow]"
        if item.required and not item.ready:
            status = "[red]failed[/red]"
        table.add_row(name, status, item.details or "-")

    Console().print(table)

