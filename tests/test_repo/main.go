
package main

import (
    "fmt"
)

// Person represents a person with basic information
type Person struct {
    Name    string
    Age     int
}

// Greet returns a greeting from the person
func (p *Person) Greet() string {
    return fmt.Sprintf("Hello, my name is %s and I am %d years old", p.Name, p.Age)
}

// NewPerson creates a new person with the given information
func NewPerson(name string, age int) *Person {
    return &Person{
        Name: name,
        Age:  age,
    }
}

func main() {
    p := NewPerson("John", 30)
    fmt.Println(p.Greet())
}
