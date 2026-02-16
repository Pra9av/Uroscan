import { Injectable } from '@angular/core';
import { CanActivate, Router } from '@angular/router';

@Injectable({
  providedIn: 'root'
})
export class AuthGuard implements CanActivate {

  constructor(private router: Router) {}

  canActivate(): boolean {

    const token = localStorage.getItem('token');
    const role = localStorage.getItem('role');

    if (!token) {
      this.router.navigate(['/']);
      return false;
    }

    // Allow access only if role exists
    if (role === 'doctor' || role === 'technician') {
      return true;
    }

    this.router.navigate(['/']);
    return false;
  }
}
