import os

bind = '0.0.0.0:5000'
workers = 1
threads = 2
reload = True
template_dir = 'web/templates'
extra_files = [os.path.join(template_dir, f) for f in os.listdir(template_dir)]
reload_extra_files = extra_files
