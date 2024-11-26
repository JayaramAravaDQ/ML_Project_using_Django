# backend/server/apps/endpoints/serializers.py

from rest_framework import serializers
from .models import MLInputData

class MLInputDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = MLInputData
        fields = '__all__'
