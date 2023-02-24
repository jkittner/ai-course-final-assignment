from werkzeug.middleware.proxy_fix import ProxyFix

from web.app import app

app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
