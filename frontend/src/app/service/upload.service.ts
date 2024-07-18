import {Injectable} from '@angular/core';
import {HttpClient} from "@angular/common/http";
import {Observable} from "rxjs";
import {Upload} from "../modal/upload";
import {RespostaUpload} from "../modal/resposta-upload";

@Injectable({
  providedIn: 'root'
})
export class UploadService {

  url = 'http://localhost:8086/upload/file';

  constructor(private http: HttpClient) { }

  uploadFile(upload: Upload): Observable<RespostaUpload> {

    const formData = new FormData();
    if (upload.file_path instanceof File)
      formData.append('file', upload.file_path);

    formData.append('description', upload.description);
    return this.http.post<RespostaUpload>(this.url, formData);
  }
}
