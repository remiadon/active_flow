def dyn_import(name):
    components = name.split('.')
    mod = __import__(".".join(components[:-1]))
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod