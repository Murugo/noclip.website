import { vec2, vec3, mat4 } from "gl-matrix";
import ArrayBufferSlice from "../ArrayBufferSlice";
import { GSPixelStorageFormat, GSMemoryMap, gsMemoryMapNew, gsMemoryMapUploadImage, gsMemoryMapReadImagePSMT4_PSMCT32, gsMemoryMapReadImagePSMT8_PSMCT32 } from "../Common/PS2/GS";
import { AABB } from "../Geometry";

export interface SilentHill3Map {
    meshGroups: MeshGroup[]
    textures: SilentHill3Texture[]
    dummyTexture: SilentHill3Texture
    meshTransforms: MeshTransform[]
}

export interface MeshGroup {
    imageSource: ImageSource;
    imageIndex: number;
    meshCount: number;
    meshes: Mesh[];
}

interface Mesh {
    texturePaletteIndex: number;
    translucent: boolean;
    submeshes: Submesh[];
}

interface Submesh {
    shapes: Shape[];
}

interface Shape {
    transformIndex: number;
    vtx: vec3[];
    vn: vec3[];
    vcol: vec3[];
    uv: vec2[];
    ind: number[];
}

interface MeshTransform {
    index: number;
    transform: mat4;
    aabb: AABB
}

export function* SubmeshIterator(meshGroups: MeshGroup[]) {
    for (const meshGroup of meshGroups) {
        for (const mesh of meshGroup.meshes) {
            for (const submesh of mesh.submeshes) {
                yield {meshGroup, mesh, submesh}
            }
        }
    }
}

export class SilentHill3Texture {
    public pixels: Uint8Array;
    public name: string;

    constructor(public width: number, public height: number, public psm: GSPixelStorageFormat, imageIndex: number, paletteIndex: number, imageSource: ImageSource) {
        this.pixels = new Uint8Array(width * height * 4);
        this.name = getTextureName(imageIndex, paletteIndex, imageSource);
    }
}

enum ImageSource {
    UNKNOWN = 0,
    GLOBAL = 1,
    TR = 2,
    LOCAL = 3
}

export function getTextureName(imageIndex: number, paletteIndex: number, imageSource: ImageSource) {
    const prefix = (imageSource === ImageSource.GLOBAL ? 'gb_' : (imageSource === ImageSource.TR ? 'tr_' : ''));
    return `tex_${prefix}${imageIndex}_${paletteIndex}`;
}

// {Image index, ImageSource} JSON -> Palette index set
type ImageRefMap = Map<string, Set<number>>;

export class SilentHill3MapParser {
    private imageRefMap: ImageRefMap = new Map;
    private gsMemoryMap: GSMemoryMap = gsMemoryMapNew();


    public parseMap(mapBuffer: ArrayBufferSlice, texGbBuffer: ArrayBufferSlice, texTrBuffer?: ArrayBufferSlice): SilentHill3Map {
        const view = mapBuffer.createDataView();
    
        // TODO: What do each of these signify?
        const groupOffsA = Math.min(view.getUint32(0x1C, true));
        const groupOffsB = Math.min(view.getUint32(0x20, true));
        const groupOffsC = Math.min(view.getUint32(0x24, true));
        const firstGroupOffs = groupOffsA > 0 ? groupOffsA : (groupOffsB > 0 ? groupOffsB : groupOffsC);
        const meshGroups = this.parseMeshGroups(view, firstGroupOffs);
        
        const texHeaderOffs = view.getUint32(0x10, true);
        
        const numGlobalImages = view.getUint16(0x42, true);
        // const numLocalImages = view.getUint16(0x44, true);
        const numTrImages = view.getUint16(0x46, true);
        let textures: SilentHill3Texture[] = [];
        if (texTrBuffer) {
            textures = textures.concat(this.processTextures(texTrBuffer, /*offset=*/0, /*baseImageIndex=*/0, ImageSource.TR));
        }
        textures = textures.concat(this.processTextures(texGbBuffer, /*offset=*/0, /*baseImageIndex=*/0, ImageSource.GLOBAL));
        textures = textures.concat(this.processTextures(mapBuffer, texHeaderOffs, /*baseImageIndex=*/numTrImages + numGlobalImages, ImageSource.LOCAL));
        
        const dummyTexture = this.createDummyTexture();

        const meshTransforms = this.parseMeshTransforms(view, view.getUint32(0xC, true));
    
        return { meshGroups, textures, dummyTexture, meshTransforms };
    }

    private parseMeshGroups(view: DataView, startOffs: number): MeshGroup[] {
        const meshGroups: MeshGroup[] = [];
        let offs = startOffs;
        while (offs > 0) {
            const nextGroupOffs = view.getUint32(offs, true);
            const imageSource = view.getUint16(offs + 0x10, true);
            const imageIndex = view.getInt32(offs + 0x14, true);
            const meshCount = view.getUint16(offs + 0x18, true);
            const imageKey = JSON.stringify({imageIndex, imageSource});
            if (!this.imageRefMap.has(imageKey)) {
                this.imageRefMap.set(imageKey, new Set);
            }
            const paletteRefMap = this.imageRefMap.get(imageKey);
            const meshes = this.parseMeshes(view, offs + view.getUint16(offs + 0x4, true), paletteRefMap);
            meshGroups.push({ imageSource, imageIndex, meshCount, meshes });
            offs = nextGroupOffs;
        }
        return meshGroups;
    }

    private parseMeshes(view: DataView, startOffs: number, paletteIndexSet: Set<number> | undefined): Mesh[] {
        const meshes: Mesh[] = [];
        let offs = startOffs;
        while (offs > 0) {
            const nextMeshOffs = view.getUint32(offs, true);
            const texturePaletteIndex = view.getUint16(offs + 0x10, true);
            const translucent = view.getUint16(offs + 0x16, true) === 1;
            if (paletteIndexSet) {
                paletteIndexSet.add(texturePaletteIndex);
            }
            const submeshes = this.parseSubmeshes(view, offs + view.getUint16(offs + 0x4, true));
            meshes.push({ texturePaletteIndex, translucent, submeshes });
            offs = nextMeshOffs;
        }
        return meshes;
    }

    private parseSubmeshes(view: DataView, startOffs: number): Submesh[] {
        const submeshes: Submesh[] = [];
        let offs = startOffs;
        while (offs > 0) {
            const nextSubmeshOffs = view.getUint32(offs, true);
            // Submesh header data is currently unknown.
            const shapes = this.parseShapes(view, offs + view.getUint16(offs + 0x4, true));
            submeshes.push({ shapes });
            offs = nextSubmeshOffs;
        }
        return submeshes;
    }

    private parseShapes(view: DataView, startOffs: number): Shape[] {
        const shapes: Shape[] = [];
        let offs = startOffs;
        while (offs > 0) {
            const nextShapeOffs = view.getUint32(offs, true);
            const numVertices = view.getUint32(offs + 0x10, true);
            const transformIndex = view.getUint32(offs + 0x14, true);
            offs += view.getUint32(offs + 0x4, true);
    
            const vtx: vec3[] = [];
            const vn: vec3[] = [];
            const vcol: vec3[] = [];
            const uv: vec2[] = [];
            const ind: number[] = [];
            let reverse = false;
            for (let i = 0; i < numVertices; i++) {
                const vertOffs = offs + i * 0x10;
                vtx.push(vec3.fromValues(
                    view.getInt16(vertOffs, true),  // / 0x8000,
                    view.getInt16(vertOffs + 0x2, true),  // / 0x8000,
                    view.getInt16(vertOffs + 0x4, true)  // / 0x8000
                ));
                vn.push(vec3.fromValues(
                    (view.getInt16(vertOffs + 0x6, true) & ~0x3F) / 0x8000,
                    (view.getInt16(vertOffs + 0xC, true) & ~0x3F) / 0x8000,
                    (view.getInt16(vertOffs + 0xE, true) & ~0x3F) / 0x8000
                ));
                vcol.push(vec3.fromValues(
                    (view.getInt16(vertOffs + 0x6, true) & 0x3F) / 0x20,
                    (view.getInt16(vertOffs + 0xC, true) & 0x3F) / 0x20,
                    (view.getInt16(vertOffs + 0xE, true) & 0x3F) / 0x20
                ));
                const u = view.getInt16(vertOffs + 0x8, true) / 0x8000;
                const v = view.getInt16(vertOffs + 0xA, true) / 0x8000;
                uv.push(vec2.fromValues(u, v));
                const flag = view.getUint8(vertOffs + 0x8) & 1;
                if (flag === 0) {
                    if (reverse) {
                        ind.push(i, i - 1, i - 2);
                    } else {
                        ind.push(i - 2, i - 1, i);
                    }
                }
                reverse = !reverse
            }
            shapes.push({ transformIndex, vtx, vn, vcol, uv, ind });
            offs = nextShapeOffs;
        }
        return shapes;
    }

    private processTextures(buffer: ArrayBufferSlice, offset: number, baseImageIndex: number, imageSource: ImageSource): SilentHill3Texture[] {
        const textures: SilentHill3Texture[] = [];
        const view = buffer.createDataView();
        const numImages = view.getUint32(offset + 0x14, true);
        let offs = offset + view.getUint32(offset + 0x8, true);
        for (let i = 0; i < numImages; i++) {
            const dataSize = view.getUint32(offs + 0x10, true);
            const dataOffs = offs + view.getUint16(offs + 0x14, true);
            const psm = view.getUint8(offs + 0x19);
            const hasClut = psm === GSPixelStorageFormat.PSMT4 || psm === GSPixelStorageFormat.PSMT8;
            let clutSize = 0;
            if (hasClut) {
                clutSize = view.getUint32(dataOffs + dataSize, true);
            }
            const nextOffs = dataOffs + dataSize + clutSize + 0x30;
            const imageIndex = (i === numImages - 1 && imageSource === ImageSource.GLOBAL) ? -1 : i + baseImageIndex;
            const imageKey = JSON.stringify({imageIndex, imageSource});
            if (!this.imageRefMap.has(imageKey)) {
                offs = nextOffs;
                continue;
            }
    
            const width = view.getUint16(offs + 0x8, true);
            const height = view.getUint16(offs + 0xA, true);
            const hFactor = view.getUint8(offs + 0x1A);
            const wFactor = hFactor > 0 ? 1 : 0;
            const dbw = width >> 6 >> wFactor;
            const rrw = width >> wFactor;
            const rrh = height >> hFactor;
            gsMemoryMapUploadImage(this.gsMemoryMap, GSPixelStorageFormat.PSMCT32, /*dbp=*/0x0, dbw, /*dsax=*/0, /*dsay=*/0, rrw, rrh, buffer.subarray(dataOffs, dataSize));
            if (hasClut) {
                const clutHeaderOffs = dataOffs + dataSize;
                const clutDataOffs = clutHeaderOffs + 0x30;
                const clutWidth = view.getUint8(clutHeaderOffs + 0xE);
                const clutHeight = Math.floor(clutSize / (clutWidth * 4));
                gsMemoryMapUploadImage(this.gsMemoryMap, GSPixelStorageFormat.PSMCT32, /*dbp=*/0x3640, /*dbw=*/1, /*dsax=*/0, /*dsay=*/0, clutWidth, clutHeight, buffer.subarray(clutDataOffs, clutSize));
            }
    
            const paletteIndexSet = this.imageRefMap.get(imageKey);
            if (paletteIndexSet) {
                var self = this;
                paletteIndexSet.forEach(function(paletteIndex: number) {
                    const texture = new SilentHill3Texture(width, height, psm, imageIndex, paletteIndex, imageSource);
                    // TODO: Support PSMCT32 image download
                    const cbp = 0x3640 + paletteIndex * 0x4;
                    if (psm == GSPixelStorageFormat.PSMT4) {
                        gsMemoryMapReadImagePSMT4_PSMCT32(texture.pixels, self.gsMemoryMap, /*dbp=*/0x0, /*dbw=*/width >> 6, /*rrw=*/width, /*rrh=*/height, cbp, /*csa=*/0, /*alphaReg=*/-1);
                    } else if (psm == GSPixelStorageFormat.PSMT8) {
                        gsMemoryMapReadImagePSMT8_PSMCT32(texture.pixels, self.gsMemoryMap, /*dbp=*/0x0, /*dbw=*/width >> 6, /*rrw=*/width, /*rrh=*/height, cbp, /*alphaReg=*/-1);
                    } else {
                        console.warn(`Image download unsupported: ${psm}`);
                    }
                    textures.push(texture);
                });
            }
            offs = nextOffs;
        }
        return textures;
    }
    
    private createDummyTexture(): SilentHill3Texture {
        const texture = new SilentHill3Texture(/*width=*/64, /*height=*/64, /*psm=*/19, /*imageIndex=*/99, /*paletteIndex=*/99, ImageSource.GLOBAL);
        for (let i = 0; i < 64 * 64 * 4; i++) {
            texture.pixels[i * 4] = 255
            texture.pixels[i * 4 + 2] = 255
            texture.pixels[i * 4 + 3] = 255
        }
        return texture;
    }

    private parseMeshTransforms(view: DataView, offset: number): MeshTransform[] {
        const meshTransforms : MeshTransform[] = [];
        let inverseFirstTransform = mat4.create();
        let offs = offset;

        // Temporary default transform.
        meshTransforms[0] = {index: 0, transform: mat4.create(), aabb: new AABB(0, 0, 0, 0, 0, 0)};

        while (offs > 0) {
            let nextOffs = view.getUint32(offs, true);
            if (nextOffs > 0) {
                nextOffs += offs;
            }
            const index = view.getUint32(offs + 0x10, true);
            offs += view.getUint32(offs + 0x4, true);

            const mVal = new Float32Array(16);
            for (let i = 0; i < 16; i++) {
                mVal[i] = view.getFloat32(offs + i * 0x4, true);
            }
            let transform = mat4.fromValues(
                mVal[0], mVal[1], mVal[2], mVal[3],
                mVal[4], mVal[5], mVal[6], mVal[7],
                mVal[8], mVal[9], mVal[10], mVal[11],
                mVal[12], mVal[13], mVal[14], mVal[15]
            );
            if (index == 1) {
                // Use first transform as world origin. Meshes may be transposed at locations such as
                // (-60000.0, 0.0, -20000.0). Doing this hides these egregious offsets in the viewer.
                // TODO: This will not be ideal once noclip supports displaying multiple maps together,
                // such as for the town of Silent Hill.
                mat4.invert(inverseFirstTransform, transform);
                transform = mat4.create();
            } else {
                mat4.multiply(transform, inverseFirstTransform, transform);
            }
            offs += 0x40;
            const aabb = new AABB(
                view.getFloat32(offs + 0x70, true), view.getFloat32(offs + 0x74, true), view.getFloat32(offs + 0x78, true),
                view.getFloat32(offs + 0x00, true), view.getFloat32(offs + 0x04, true), view.getFloat32(offs + 0x08, true));

            meshTransforms[index] = {index, transform, aabb};
            offs = nextOffs;
        }
        return meshTransforms;
    }
}
