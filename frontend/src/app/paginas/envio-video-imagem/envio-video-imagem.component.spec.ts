import { ComponentFixture, TestBed } from '@angular/core/testing';

import { EnvioVideoImagemComponent } from './envio-video-imagem.component';

describe('EnvioVideoImagemComponent', () => {
  let component: EnvioVideoImagemComponent;
  let fixture: ComponentFixture<EnvioVideoImagemComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [EnvioVideoImagemComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(EnvioVideoImagemComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
