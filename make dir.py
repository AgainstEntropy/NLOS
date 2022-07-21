import os

classes = ['Lying down',
           'Kneel to stand',
           'Squat',
           'Jumping jacks',
           'Side shuffle',
           'Turn around',
           'Being hit',
           'Yelling',
           'Strafing',
           'Stand to crouch',
           'Hanging',
           'Throw',
           'Cartwheel',
           'Spin',
           'Hip hop']


clas_root = 'G:/Blender/blank_wall/character_actions/unseen/actions_skeleton'
for cls in classes:
    cls_dir = os.path.join(clas_root, cls)
    os.makedirs(cls_dir, exist_ok=True)
