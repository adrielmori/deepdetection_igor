import {Component} from '@angular/core';
import {NgbAccordionButton} from "@ng-bootstrap/ng-bootstrap";
import {Upload, UploadTipoArquivo} from "../../modal/upload";
import {FormsModule} from "@angular/forms";
import {NgClass, NgForOf, NgIf} from "@angular/common";
import {UploadService} from "../../service/upload.service";
import {catchError, EMPTY} from "rxjs";
import {ToastrService} from "ngx-toastr";
import {RespostaUpload} from "../../modal/resposta-upload";
import {DownloadService} from "../../service/download.service";

@Component({
  selector: 'app-envio-video-imagem',
  standalone: true,
  imports: [
    NgbAccordionButton,
    FormsModule,
    NgClass,
    NgForOf,
    NgIf,
  ],
  templateUrl: './envio-video-imagem.component.html',
  styleUrl: './envio-video-imagem.component.scss'
})
export class EnvioVideoImagemComponent {

  nomeArquivo: string = ''

  upload: Upload = {
    description: UploadTipoArquivo.select,
    file_path: ''
  }

  mudouArquivo(event: any) {
    this.upload.file_path = event.target.files[0];
    if (typeof this.upload.file_path !== "string") {
      this.nomeArquivo = this.upload.file_path.name;
    }
  }

  constructor(private uploadService: UploadService, private toastr: ToastrService,  private downloadService: DownloadService) {}

  enviar() {
    this.uploadService.uploadFile(this.upload)
      .pipe(catchError(err => {
        this.toastr.error(err.error.message, err.statusText);
        return EMPTY;
      }))
      .subscribe((res: RespostaUpload) => {
        this.toastr.success(`${res.score}` , res.target);
        this.downloadService.downloadFile(this.nomeArquivo)
      })
  }

  protected readonly Object = Object;
  protected readonly UploadTipoArquivo = UploadTipoArquivo;
}
