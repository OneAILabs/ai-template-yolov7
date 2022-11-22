import os

# notebook: <module 'jupyter_server' from '/root/miniconda3/lib/python3.9/site-packages/jupyter_server/__init__.py'>
# c: <class 'traitlets.config.loader.Config'>


# without this, you can "log out" or "shut down" by the following way, 
#
# CaseA: [Jupyter UI][menu] File > Log out     <--- disable it
# CaseB: [Jupyter UI][menu] File > Shut down   <--- disable it
#
# If CaseA:
#  -> [menu] File > Log out 
#  -> redirected to the infra-login page.
#  -> log in again (OK)
#
# If CaseB:
#  -> [menu] File > Shut down
#  -> (After shutting down, you won't be able to enter the jupyter UI)
#  -> launch again (in the [Notebook Service Details] page)
#  -> you will get the following error:
#     ```
#     {
#       "message":"An invalid response was received from the upstream server"
#     }
#     ```
#  -> you need to stop & start the notebook service
#
# @see https://jupyter-server.readthedocs.io/en/latest/other/full-config.html
#
c.ServerApp.quit_button = False

# without 'TOKEN', you are required to enter a password or token
# [Jupyter UI] Password or token: [________]  [Log in]
# [Jupyter UI] Token authentication is enabled
#
c.ServerApp.token = os.environ.get('TOKEN', '')

# if TOKEN is unset, check PASSWORD
if not c.ServerApp.token:
    password = os.environ.get('PASSWORD', '')
    if password:
        from notebook.auth import passwd
        c.ServerApp.password = passwd(password)


# withuot 'BASE_URL', it will cause: 
# - {"message":"no Route matched with those values"}
#
# default URL: 
# https://your_host/lab?
#
# after applying the following config:
# BASE_URL=/jupyter/88b70935-48f1-43bd-80a0-24b46e3858fb/parabricks-1/
#
# the URL becomes: 
# https://your_host/jupyter/88b70935-48f1-43bd-80a0-24b46e3858fb/parabricks-1/lab?
#
c.ServerApp.base_url = os.environ.get('BASE_URL', '/')
