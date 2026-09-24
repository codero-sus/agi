def register(api):
    @api.skill(
        "take_note",
        "Store a short note the user wants remembered.",
        pattern=r"^(note:|take a note|make a note)",
    )
    def take_note(ctx, text=""):
        msg = ctx.get("user") or text
        body = msg.split(":", 1)[-1].strip() if ":" in msg else msg
        mem = ctx.get("memory")
        if mem is not None:
            mem.add_fact("user", "note", body, 0.9)
        return f"Noted: {body}"
