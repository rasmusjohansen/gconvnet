group_registry = {}
action_registry = {}

def register_group(group, name):
    if name in group_registry:
        raise ValueError(name + ' already registered as group')

    group_registry[name] = group

def register_action(action,name):
    if name in action_registry:
        raise ValueError(name + ' already registered as action')

    action_registry[name] = action

def get_group(name):
    if not isinstance(name,str):
        return name
    
    if name not in group_registry:
        raise KeyError(name + ' not found in group registry')
    
    return group_registry[name]

def get_action(name):
    if not isinstance(name,str):
        return name
    
    if name not in action_registry:
        raise KeyError(name + ' not found in action registry')

    return action_registry[name]
