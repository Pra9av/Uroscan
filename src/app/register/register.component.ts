import { Component } from '@angular/core';
import { Router } from '@angular/router';


@Component({
  selector: 'app-register',
  standalone: false,
  templateUrl: './register.component.html',
  styleUrl: './register.component.scss'
})
export class RegisterComponent {
  name: string = '';
email: string = '';
password: string = '';
role: string = '';
  constructor(private router: Router) {}

  register() {
  const userData = {
    name: this.name,
    email: this.email,
    password: this.password,
    role: this.role
  };

  fetch("http://localhost:5000/api/auth/register", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(userData)
  })
  .then(res => res.json())
  .then(data => {
    if (data.message === "User registered successfully") {
      alert("Registration successful");
      this.router.navigate(['/']); // go to login
    } else {
      alert(data.message);
    }
  })
  .catch(err => console.error(err));
}
}