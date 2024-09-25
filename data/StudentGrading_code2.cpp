#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>  // For random numbers
#include <ctime>    // For seeding random numbers

// Class to represent a subject and its grade
class Subject {
public:
    std::string name;
    double grade;

    Subject(const std::string& name, double grade) : name(name), grade(grade) {}

    void displaySubject() const {
        std::cout << "Subject: " << name << ", Grade: " << grade << std::endl;
    }
};

// Class to represent a student
class Student {
public:
    std::string name;
    std::vector<Subject> subjects;

    Student(const std::string& name) : name(name) {}

    // Add a subject and grade for the student
    void addSubject(const std::string& subjectName, double grade) {
        subjects.emplace_back(subjectName, grade);
    }

    // Convert a percentage grade to GPA (based on 4.0 scale)
    double convertToGPA(double grade) const {
        if (grade >= 90)
            return 4.0;
        else if (grade >= 80)
            return 3.0;
        else if (grade >= 70)
            return 2.0;
        else if (grade >= 60)
            return 1.0;
        else
            return 0.0;
    }

    // Calculate the GPA for the student
    double calculateGPA() const {
        double totalGPA = 0;
        for (const auto& subject : subjects) {
            totalGPA += convertToGPA(subject.grade);
        }
        return subjects.empty() ? 0 : totalGPA / subjects.size();
    }

    // Display the student's details, subjects, and GPA
    void displayStudentDetails() const {
        std::cout << "Student Name: " << name << std::endl;
        for (const auto& subject : subjects) {
            subject.displaySubject();
        }
        double gpa = calculateGPA();
        std::cout << "GPA: " << gpa << std::endl;
    }
};

// Class to manage multiple students and their grades
class StudentGradingSystem {
public:
    std::vector<Student> students;
    std::vector<std::string> subjects = {"Math", "Science", "History", "English", "Computer Science"};

    // Add a new student to the system
    void addStudent(const std::string& studentName) {
        students.emplace_back(studentName);
    }

    // Add a random subject and grade for a specific student
    void addRandomGradesToStudent(Student& student) {
        for (const auto& subject : subjects) {
            double grade = (std::rand() % 41) + 60;  // Random grade between 60 and 100
            student.addSubject(subject, grade);
        }
    }

    // Generate and add 20 students with random grades
    void generateClassOfStudents() {
        for (int i = 1; i <= 20; ++i) {
            std::string studentName = "Student " + std::to_string(i);
            addStudent(studentName);
            addRandomGradesToStudent(students.back());
        }
    }

    // Display details of all students
    void displayAllStudents() const {
        for (const auto& student : students) {
            student.displayStudentDetails();
            std::cout << "-----------------------" << std::endl;
        }
    }
};

// Main function to test the student grading system
int main() {
    std::srand(std::time(nullptr));  // Seed for random number generation

    StudentGradingSystem system;

    // Generate and add 20 students with random grades
    system.generateClassOfStudents();

    // Display all students and their grades
    system.displayAllStudents();

    return 0;
}
