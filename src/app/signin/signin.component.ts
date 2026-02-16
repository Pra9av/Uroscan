import { Component } from '@angular/core';
import { Router } from '@angular/router';

@Component({
  selector: 'app-signin',
  standalone: false,
  templateUrl: './signin.component.html',
  styleUrl: './signin.component.scss'
})
export class SigninComponent {
    email: string = '';
  password: string = '';

  constructor(private router: Router) {}

  login() {
  const userData = {
    email: this.email,
    password: this.password
  };

  fetch("http://localhost:5000/api/auth/login", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(userData)
  })
  .then(res => res.json())
  .then(data => {
    if (data.token) {
      localStorage.setItem("token", data.token);
      localStorage.setItem("role", data.role);

      if (data.role === "doctor") {
        this.router.navigate(["/doctor"]);
      } else {
        this.router.navigate(["/technician"]);
      }
    } else {
      alert("Invalid credentials");
    }
  })
  .catch(err => console.error(err));
}

}
