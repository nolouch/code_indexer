package main

import (
	"fmt"
	"strings"
)

// Person represents a person with basic information
type Person struct {
	Name    string `json:"name"`
	Age     int    `json:"age"`
	Address string `json:"address"`
}

// Greeter defines the greeting behavior
type Greeter interface {
	Greet() string
	SetName(name string)
}

// Implement Greeter interface for Person
func (p *Person) Greet() string {
	return fmt.Sprintf("Hello, my name is %s and I am %d years old", p.Name, p.Age)
}

func (p *Person) SetName(name string) {
	p.Name = strings.TrimSpace(name)
}

// CreatePerson creates a new person with the given information
func CreatePerson(name string, age int, address string) *Person {
	return &Person{
		Name:    name,
		Age:     age,
		Address: address,
	}
}

func main() {
	person := CreatePerson("John Doe", 30, "123 Main St")
	fmt.Println(person.Greet())

	person.SetName("Jane Doe")
	fmt.Println(person.Greet())
}
