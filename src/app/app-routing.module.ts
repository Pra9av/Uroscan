import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { DashboardComponent } from './dashboard/dashboard.component';
import { TechnicianComponent } from './technician/technician.component';
import { SigninComponent } from './signin/signin.component';
import { RegisterComponent } from './register/register.component';
import { DoctorComponent } from './doctor/doctor.component';
import { AuthGuard } from './auth.guard';



const routes: Routes = [
    { path: '', component: DashboardComponent },
    { path: 'technician', component: TechnicianComponent, canActivate: [AuthGuard]  },
    { path: 'signin', component: SigninComponent },
    { path: 'register', component: RegisterComponent },
    { path: 'doctor', component: DoctorComponent, canActivate: [AuthGuard] }






];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
