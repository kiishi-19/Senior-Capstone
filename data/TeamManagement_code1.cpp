#include <iostream>
#include <vector>
#include <string>

class Task {
private:
    std::string description;
    bool isCompleted;

public:
    Task(std::string desc) : description(desc), isCompleted(false) {}

    void completeTask() {
        isCompleted = true;
    }

    std::string getDescription() const { return description; }
    bool getStatus() const { return isCompleted; }
};

class TeamMember {
private:
    std::string name;
    std::vector<Task> tasks;

public:
    TeamMember(std::string name) : name(name) {}

    void assignTask(const Task& task) {
        tasks.push_back(task);
        std::cout << "Task assigned to " << name << ": " << task.getDescription() << std::endl;
    }

    void completeTask(int index) {
        if (index >= 0 && index < tasks.size()) {
            tasks[index].completeTask();
            std::cout << "Task completed by " << name << ": " << tasks[index].getDescription() << std::endl;
        } else {
            std::cout << "Invalid task index" << std::endl;
        }
    }

    void listTasks() const {
        std::cout << "Tasks for " << name << ":\n";
        for (const auto& task : tasks) {
            std::cout << " - " << task.getDescription() << (task.getStatus() ? " [Completed]" : " [Pending]") << std::endl;
        }
    }

    std::string getName() const { return name; }
};

class Project {
private:
    std::string title;
    std::vector<TeamMember> teamMembers;

public:
    Project(std::string title) : title(title) {}

    void addTeamMember(const TeamMember& member) {
        teamMembers.push_back(member);
        std::cout << "Added team member: " << member.getName() << std::endl;
    }

    void assignTaskToMember(const std::string& memberName, const Task& task) {
        for (auto& member : teamMembers) {
            if (member.getName() == memberName) {
                member.assignTask(task);
                return;
            }
        }
        std::cout << "Team member not found" << std::endl;
    }

    void listProjectDetails() const {
        std::cout << "Project: " << title << std::endl;
        for (const auto& member : teamMembers) {
            member.listTasks();
        }
    }
};

int main() {
    Project project("AI Development");

    TeamMember alice("Alice");
    TeamMember bob("Bob");

    project.addTeamMember(alice);
    project.addTeamMember(bob);

    project.assignTaskToMember("Alice", Task("Design AI architecture"));
    project.assignTaskToMember("Bob", Task("Implement neural network"));

    project.listProjectDetails();

    alice.completeTask(0);
    project.listProjectDetails();

    return 0;
}
