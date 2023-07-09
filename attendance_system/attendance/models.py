from django.db import models

# Create your models here.
class Student(models.Model):
    name = models.CharField(max_length=100)
    student_id = models.CharField(max_length=20)
    # Add other fields as needed

    def __str__(self):
        return self.name


class Attendance(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    date = models.DateField()
    time = models.TimeField()
    # Add other fields as needed

    def __str__(self):
        return f"{self.student} - {self.date} {self.time}"
