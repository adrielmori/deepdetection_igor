import {File} from "node:buffer";

export enum UploadTipoArquivo {
  select = 'select',
  image = 'image',
  video = 'video',
}

export interface Upload {
  description: UploadTipoArquivo;
  file_path: File | string;
}
