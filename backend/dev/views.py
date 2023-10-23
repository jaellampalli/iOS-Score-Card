from django.shortcuts import render
from django.views.generic import TemplateView
from django.http import HttpResponse

# Create your views here.
class HomeView(TemplateView):
    template_name = "index.html"
    def get(self, request):
        return render(request,'index.html')
    
def api(request):
    if(request.method == 'POST'):
        print("test")

        print(request.FILES)
        
        return HttpResponse(request)
    
