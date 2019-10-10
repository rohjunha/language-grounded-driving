def destroy_actor(actor):
    if actor is not None and actor.is_alive:
        actor.destroy()
