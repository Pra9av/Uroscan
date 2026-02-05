import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { DashboardComponent } from './dashboard/dashboard.component';
import { TechnicianComponent } from './technician/technician.component';


const routes: Routes = [
    { path: '', component: DashboardComponent },
    { path: 'technician', component: TechnicianComponent }


];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
