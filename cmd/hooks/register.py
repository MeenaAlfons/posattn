from . import hooks


def register_hooks(model, hook_names, config, context):
    hook_classes = [getattr(hooks, hook_name) for hook_name in hook_names]
    for name, module in model.named_modules():
        for hook_class in hook_classes:
            if hook_class.is_applicable(name, module):
                hook = hook_class(name, module, config)
                hook = hooks.LogWhenNeededHook(hook, context)
                if hook_class.is_forward_hook():
                    module.register_forward_hook(hook)

                if hook_class.is_backward_hook():
                    module.register_backward_hook(hook)
