import { Injectable } from '@angular/core';
import {HttpClient} from "@angular/common/http";

@Injectable({
  providedIn: 'root'
})
export class DownloadService {


  url = 'http://localhost:8086/processed/';

  constructor(private http: HttpClient) { }

  downloadFile(file: string)  {
    this.http.get(this.url + file,
      { responseType: 'blob' })
      .subscribe(blob => {
        const a = document.createElement('a');
        const objectUrl = URL.createObjectURL(blob);
        a.href = objectUrl;
        a.download = file;
        a.click();
        URL.revokeObjectURL(objectUrl);
      });
  }
}
