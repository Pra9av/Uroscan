import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';


@Component({
  selector: 'app-doctor',
  standalone: false,
  templateUrl: './doctor.component.html',
  styleUrl: './doctor.component.scss'
})
export class DoctorComponent  implements OnInit {
  role: string | null = '';

  constructor(private router: Router) {}

  ngOnInit(): void {
    this.role = localStorage.getItem('role');
  }

  logout() {
    localStorage.clear();
    this.router.navigate(['/']);
  }

}
