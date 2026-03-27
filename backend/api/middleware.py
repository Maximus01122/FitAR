class AllowAllCorsMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.method == "OPTIONS":
            response = self._build_preflight_response()
        else:
            response = self.get_response(request)
        return self._apply_headers(request, response)

    def _build_preflight_response(self):
        from django.http import HttpResponse

        return HttpResponse(status=204)

    def _apply_headers(self, request, response):
        request_origin = request.headers.get("Origin")
        response["Access-Control-Allow-Origin"] = request_origin or "*"
        response["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
        response["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response["Vary"] = "Origin"
        return response
