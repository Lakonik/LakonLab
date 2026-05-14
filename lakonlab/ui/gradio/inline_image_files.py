from pathlib import Path
from urllib.parse import unquote

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


IMAGE_MIME = {
    ".webp": "image/webp",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
}


class InlineGradioImageFiles(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        url = str(request.url)
        if "/gradio_api/file=" not in url and "/file=" not in url:
            return response

        file_url = unquote(url.split("file=", 1)[1].split("?", 1)[0])
        filename = Path(file_url).name
        mime = IMAGE_MIME.get(Path(filename).suffix.lower())
        if mime is None:
            return response

        response.headers["content-type"] = mime
        response.headers["content-disposition"] = f'inline; filename="{filename}"'
        return response


def install_inline_image_file_middleware():
    from gradio import routes

    if getattr(routes.App, "_lakonlab_inline_image_files_patch", False):
        return

    create_app = routes.App.create_app

    def create_app_with_inline_image_files(*args, **kwargs):
        app = create_app(*args, **kwargs)
        if not any(middleware.cls is InlineGradioImageFiles for middleware in app.user_middleware):
            app.add_middleware(InlineGradioImageFiles)
        return app

    routes.App.create_app = staticmethod(create_app_with_inline_image_files)
    routes.App._lakonlab_inline_image_files_patch = True
