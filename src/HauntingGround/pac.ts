import { vec2, vec3, vec4, mat4 } from "gl-matrix";
import { gsMemoryMapNew, gsMemoryMapUploadImage, GSPixelStorageFormat, gsMemoryMapReadImagePSMT4_PSMCT32, gsMemoryMapReadImagePSMT8_PSMCT32 } from "../Common/PS2/GS";
import ArrayBufferSlice from "../ArrayBufferSlice";

export interface HauntingGroundStage {
    meshGroups: MeshGroup[]
    textures: Texture[]
}

export class MeshGroup {
    vtx: vec3[] = [];
    vcol: vec4[] = [];
    uv: vec2[] = [];
    ind: number[] = [];
    translucent: boolean;
    addBlendMode: boolean;
    textureIndex: number;
}

export class Texture {
    public pixels: Uint8Array;
    public name: string;

    constructor(public width: number, public height: number, public psm: GSPixelStorageFormat, index: number) {
        this.pixels = new Uint8Array(width * height * 4);
        this.name = `TEX_${index}`;
    }
}

export function parseStagePAC(buffer: ArrayBufferSlice): HauntingGroundStage {
    const view = buffer.createDataView();

    const meshGroups: MeshGroup[] = [];
    const vifSectorOffs = view.getUint32(0xC, true);
    let vifSectorEndOffs = view.getUint32(0x10, true);
    if (vifSectorEndOffs < vifSectorOffs) {
        vifSectorEndOffs = view.getUint32(0x14, true);
    }
    const vifView = buffer.createDataView(vifSectorOffs, vifSectorEndOffs - vifSectorOffs);
    let offs = vifView.getUint32(0, true);
    while (offs < vifView.byteLength) {
        const numVertices = vifView.getInt32(offs, true);
        let textureIndex = vifView.getInt32(offs + 0x4, true);
        let translucent = (vifView.getUint32(offs + 0x8, true) & 1) > 0;
        const addBlendMode = (vifView.getUint8(offs + 0xD) & 1) > 0;
        const flags = vifView.getUint32(offs + 0x8, true);

        // Quick hack to get sky to render correctly in ST_000 and ST_10B.
        if (vifView.getUint8(offs + 0x9) > 0) {
            translucent = true;
        }

        offs += 0x10;
        if (numVertices <= 0) {
            continue;
        }
        const meshGroup = new MeshGroup;
        meshGroup.textureIndex = textureIndex;
        meshGroup.translucent = translucent;
        meshGroup.addBlendMode = addBlendMode;
        meshGroups.push(meshGroup);
        
        const mVal = new Float32Array(16);
        for (let i = 0; i < 16; i++) {
            mVal[i] = vifView.getFloat32(offs + i * 0x4, true);
        }
        const m = mat4.fromValues(
            mVal[0], mVal[4], mVal[8], mVal[12],
            mVal[1], mVal[5], mVal[9], mVal[13],
            mVal[2], mVal[6], mVal[10], mVal[14],
            mVal[3], mVal[7], mVal[11], mVal[15]
        );
        offs += 0x40;

        for (let i = 0; i < numVertices; i++) {
            meshGroup.uv.push(vec2.fromValues(
                vifView.getFloat32(offs + i * 0x8, true),
                vifView.getFloat32(offs + i * 0x8 + 0x4, true)
            ))
        }
        offs += Math.ceil(numVertices * 0x8 / 0x10) * 0x10;

        for (let i = 0; i < numVertices; i++) {
            meshGroup.vcol.push(vec4.fromValues(
                vifView.getUint8(offs + i * 0x4) / 0x100,
                vifView.getUint8(offs + i * 0x4 + 0x1) / 0x100,
                vifView.getUint8(offs + i * 0x4 + 0x2) / 0x100,
                vifView.getUint8(offs + i * 0x4 + 0x3) / 0x80
            ))
        }
        offs += Math.ceil(numVertices * 0x4 / 0x10) * 0x10;

        let r = false;
        for (let i = 0; i < numVertices; i++) {
            let v = vec4.fromValues(
                vifView.getFloat32(offs + i * 0x10, true),
                vifView.getFloat32(offs + i * 0x10 + 0x4, true),
                vifView.getFloat32(offs + i * 0x10 + 0x8, true), 1);
            vec4.transformMat4(v, v, m);
            meshGroup.vtx.push(vec3.fromValues(v[0], v[1], v[2]));
            const w = vifView.getUint16(offs + i * 0x10 + 0xC, true);
            if (w == 0x8000) {
                r = false;
                continue;
            }
            if (r) {
                meshGroup.ind.push(i, i - 1, i - 2);
            } else {
                meshGroup.ind.push(i - 2, i - 1, i);
            }
            r = !r;
        }
        offs += numVertices * 0x10;
    }

    const textures: Texture[] = []
    const gsMemoryMap = gsMemoryMapNew();
    const texSectorOffs = view.getUint32(0x24, true);
    const texView = buffer.createDataView(texSectorOffs);
    const texCount = texView.getUint32(0, true);
    for (let i = 0; i < texCount; i++) {
        const dpsm = texView.getUint32(i * 0x10 + 0x10, true);
        const width = texView.getUint16(i * 0x10 + 0x14, true);
        const height = texView.getUint16(i * 0x10 + 0x16, true);
        const dataSize = texView.getUint16(i * 0x10 + 0x18, true) << 4;
        const clutSize = texView.getUint16(i * 0x10 + 0x1A, true) << 4;
        const dataOffs = texView.getUint32(i * 0x10 + 0x1C, true);
        const clutOffs = dataOffs + dataSize;

        gsMemoryMapUploadImage(gsMemoryMap, dpsm, /*dbp=*/0x8, /*dbw*/width >> 6, /*dsax=*/0, /*dsay=*/0, /*rrw=*/width, /*rrh=*/height, buffer.subarray(texSectorOffs + dataOffs + (i + 1) * 0x10, dataSize));
        gsMemoryMapUploadImage(gsMemoryMap, /*dpsm=*/GSPixelStorageFormat.PSMCT32, /*dbp=*/0, /*dbw*/1, /*dsax=*/0, /*dsay=*/0, /*rrw=*/0x10, /*rrh=*/clutSize / 0x40, buffer.subarray(texSectorOffs + clutOffs + (i + 1) * 0x10, clutSize));
        const texture = new Texture(width, height, dpsm, i);
        if (dpsm == GSPixelStorageFormat.PSMT4) {
            gsMemoryMapReadImagePSMT4_PSMCT32(texture.pixels, gsMemoryMap, /*dbp=*/0x8, /*dbw=*/width >> 6, /*rrw=*/width, /*rrh=*/height, /*cbp=*/0, /*csa=*/0, /*alphaReg=*/-1);
        } else if (dpsm == GSPixelStorageFormat.PSMT8) {
            gsMemoryMapReadImagePSMT8_PSMCT32(texture.pixels, gsMemoryMap, /*dbp=*/0x8, /*dbw=*/width >> 6, /*rrw=*/width, /*rrh=*/height, /*cbp=*/0, /*alphaReg=*/-1);
        }
        textures.push(texture);
    }

    return {meshGroups, textures};
}
