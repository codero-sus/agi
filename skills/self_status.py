def register(api):
    @api.skill(
        "self_status",
        "Summarize who I am and how I am growing.",
        pattern=r"how (are|have) you (improving|growing|learning)",
    )
    def self_status(ctx, text=""):
        ident = ctx.get("identity")
        counts = ctx.get("counts") or {}
        if ident is None:
            return "online"
        snap = ident.snapshot()
        return (
            f"{snap['name']} · turns {snap['turns']} · cycles {snap['cycles']} · "
            f"constitution v{snap['constitution_version']} · "
            f"memories {counts.get('episodes', 0)} · facts {counts.get('facts', 0)} · "
            f"lessons {counts.get('lessons', 0)}"
        )
