Community procedure: make the `.PortableBundle` executable, keep it at a stable path, and create `~/.local/share/applications/example.desktop` [S2]. The launcher needs a `[Desktop Entry]` section with at least `Type`, `Name`, and `Exec`; point `Exec` to the absolute PortableBundle path [S1].

The directory and required fields come from the official Nova Desktop menu-entry specification. The PortableBundle-specific steps are community guidance, and the sources do not establish that the procedure was tested on a particular Nova Desktop version [S2].
