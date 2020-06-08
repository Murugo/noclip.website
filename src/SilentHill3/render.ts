import * as UI from '../ui';
import * as Viewer from '../viewer';

// @ts-ignore
import program_glsl from './program.glsl';
import { ViewerRenderInput, SceneGfx } from "../viewer";
import { GfxRenderInstManager, executeOnPass } from "../gfx/render/GfxRenderer";
import { BasicRenderTarget, transparentBlackFullClearRenderPassDescriptor } from "../gfx/helpers/RenderTargetHelpers";
import { GfxRenderDynamicUniformBuffer } from "../gfx/render/GfxRenderDynamicUniformBuffer";
import { GfxDevice, GfxHostAccessPass, GfxRenderPass, GfxBindingLayoutDescriptor, makeTextureDescriptor2D, GfxFormat, GfxProgram, GfxMegaStateDescriptor, GfxColorWriteMask, GfxBlendMode, GfxBlendFactor, GfxSampler, GfxInputState, GfxInputLayout, GfxBuffer, GfxWrapMode, GfxTexFilterMode, GfxMipFilterMode, GfxBufferUsage, GfxVertexAttributeDescriptor, GfxInputLayoutBufferDescriptor, GfxVertexBufferFrequency, GfxCullMode, GfxCompareMode } from "../gfx/platform/GfxPlatform";
import { SilentHill3Map, SilentHill3Texture, SubmeshIterator, getTextureName } from "./map"
import { psmToString } from '../Common/PS2/GS';
import { LoadedTexture, TextureHolder, TextureMapping } from '../TextureHolder';
import { DeviceProgram } from '../Program';
import { mat4, vec3 } from 'gl-matrix';
import { nArray } from '../util';
import { fillMatrix4x4, fillMatrix4x3, fillVec4 } from '../gfx/helpers/UniformBufferHelpers';
import { makeStaticDataBuffer } from '../gfx/helpers/BufferHelpers';
import { reverseDepthForCompareMode } from '../gfx/helpers/ReversedDepthHelpers';
import { CameraController } from '../Camera';

class SilentHill3Program extends DeviceProgram {
    public static a_Position = 0;
    public static a_Normal = 1;
    public static a_Color = 2;
    public static a_TexCoord = 3;

    public static ub_SceneParams = 0;
    public static ub_DrawParams = 1;

    private static program = program_glsl;
    public both = SilentHill3Program.program;
}

const enum RenderPass {
    MAIN = 0x1
}

const bindingLayouts: GfxBindingLayoutDescriptor[] = [
    { numUniformBuffers: 2, numSamplers: 1 },
];

interface DrawCall {
    firstIndex: number;
    indexCount: number;
    textureIndex: number;
    translucent: boolean;
    transform: mat4;
}

const worldScale = 0.1;
let worldMatrix = mat4.fromScaling(mat4.create(), vec3.fromValues(worldScale, worldScale, worldScale));
mat4.rotateX(worldMatrix, worldMatrix, Math.PI);
let modelMatrixScratch = mat4.create();
class DrawCallInstance {
    private gfxProgram: GfxProgram | null = null;
    private program!: DeviceProgram;
    private textureMappings = nArray(1, () => new TextureMapping());
    private megaStateFlags: Partial<GfxMegaStateDescriptor>;

    private vertexColorsEnabled = true;
    private texturesEnabled = true;

    constructor(private sceneData: SceneData, private drawCall: DrawCall, private drawCallIndex: number, textureHolder: TextureHolder<any>) {
        this.createProgram();

        this.megaStateFlags = {};
        if (drawCall.translucent) {
            this.megaStateFlags.attachmentsState = [
                {
                    colorWriteMask: GfxColorWriteMask.ALL,
                    rgbBlendState: {
                        blendMode: GfxBlendMode.ADD,
                        blendSrcFactor: GfxBlendFactor.SRC_ALPHA,
                        blendDstFactor: GfxBlendFactor.ONE_MINUS_SRC_ALPHA,
                    },
                    alphaBlendState: {
                        blendMode: GfxBlendMode.ADD,
                        blendSrcFactor: GfxBlendFactor.ONE,
                        blendDstFactor: GfxBlendFactor.ZERO,
                    },
                }
            ];
        }

        const textureMapping = this.textureMappings[0];
        const textureName = sceneData.textureNames[drawCall.textureIndex];
        if (drawCall.textureIndex >= 0 && textureHolder.hasTexture(textureName)) {
            textureHolder.fillTextureMapping(textureMapping, textureName);
        } else {
            textureHolder.fillTextureMapping(textureMapping, sceneData.dummyTextureName);
        }
        textureMapping.gfxSampler = sceneData.sampler;
    }

    public prepareToRender(device: GfxDevice, renderInstManager: GfxRenderInstManager, viewerInput: Viewer.ViewerRenderInput) {
        const renderInst = renderInstManager.newRenderInst();
        renderInst.setInputLayoutAndState(this.sceneData.inputLayout, this.sceneData.inputState);
        renderInst.sortKey = this.drawCallIndex;
        if (this.drawCall.translucent) {
            renderInst.sortKey += 1000;
        }
        if (this.gfxProgram === null) {
            this.gfxProgram = renderInstManager.gfxRenderCache.createProgram(device, this.program);
        }
        renderInst.setGfxProgram(this.gfxProgram);
        renderInst.setMegaStateFlags(this.megaStateFlags);
        renderInst.setSamplerBindingsFromTextureMappings(this.textureMappings);
        renderInst.drawIndexes(this.drawCall.indexCount, this.drawCall.firstIndex);

        mat4.multiply(modelMatrixScratch, worldMatrix, this.drawCall.transform);

        let offs = renderInst.allocateUniformBuffer(SilentHill3Program.ub_DrawParams, 32);
        const mapped = renderInst.mapUniformBufferF32(SilentHill3Program.ub_DrawParams);
        offs += fillMatrix4x4(mapped, offs, modelMatrixScratch);
        offs += fillMatrix4x3(mapped, offs, viewerInput.camera.viewMatrix);
    }

    public setVertexColorsEnabled(v: boolean): void {
        this.vertexColorsEnabled = v;
        this.createProgram();
    }

    public setTexturesEnabled(v: boolean): void {
        this.texturesEnabled = v;
        this.createProgram();
    }

    private createProgram(): void {
        const program = new SilentHill3Program();
        if (this.vertexColorsEnabled) {
            program.defines.set('USE_VERTEX_COLOR', '1');
        }
        if (this.texturesEnabled) {
            program.defines.set('USE_TEXTURE', '1');
        }
        if (!this.drawCall.translucent) {
            program.defines.set(`USE_ALPHA_MASK`, '1');
        }
        this.gfxProgram = null;
        this.program = program;
    }
}

class SceneData {
    public vertexBuffer: GfxBuffer;
    public indexBuffer: GfxBuffer;
    public inputLayout: GfxInputLayout;
    public inputState: GfxInputState;

    public drawCalls: DrawCall[] = [];
    public sampler: GfxSampler;

    public textureNames: string[] = [];
    public dummyTextureName: string;

    private vertices = 0;
    private indices = 0;

    constructor(device: GfxDevice, map: SilentHill3Map) {
        for (let i = 0; i < map.textures.length; i++) {
            this.textureNames[i] = map.textures[i].name;
        }
        this.dummyTextureName = map.dummyTexture.name;
        this.sampler = this.createSampler(device);

        this.createDrawCalls(map);
        this.buidBuffersAndInputState(device, map);
    }

    private createSampler(device: GfxDevice) {
        return device.createSampler({
            wrapS: GfxWrapMode.CLAMP,
            wrapT: GfxWrapMode.CLAMP,
            minFilter: GfxTexFilterMode.BILINEAR,
            magFilter: GfxTexFilterMode.BILINEAR,
            mipFilter: GfxMipFilterMode.NO_MIP,
            minLOD: 0, maxLOD: 0,
        });
    }

    private createDrawCalls(map: SilentHill3Map) {
        for (let context of SubmeshIterator(map.meshGroups)) {
            const textureIndex = this.textureNames.indexOf(getTextureName(context.meshGroup.imageIndex, context.mesh.texturePaletteIndex, context.meshGroup.imageSource));
            const translucent = context.mesh.translucent;
            let firstIndex = this.indices;
            let lastTransformIndex = -1;
            for (let shape of context.submesh.shapes) {
                if (lastTransformIndex >= 0 && shape.transformIndex != lastTransformIndex) {
                    const indexCount = this.indices - firstIndex;
                    const transformIndex = lastTransformIndex in map.meshTransforms ? lastTransformIndex : 0;
                    this.drawCalls.push({ firstIndex, indexCount, textureIndex, translucent, transform: map.meshTransforms[transformIndex].transform });
                    firstIndex = this.indices;
                }
                this.vertices += shape.vtx.length;
                this.indices += shape.ind.length;
                lastTransformIndex = shape.transformIndex;
            }
            const indexCount = this.indices - firstIndex;
            const transformIndex = lastTransformIndex in map.meshTransforms ? lastTransformIndex : 0;
            this.drawCalls.push({ firstIndex, indexCount, textureIndex, translucent, transform: map.meshTransforms[transformIndex].transform });
        }
    }

    private buidBuffersAndInputState(device: GfxDevice, map: SilentHill3Map) {
        const vBuffer = new Float32Array(this.vertices * 12);
        const iBuffer = new Uint32Array(this.indices);
        let vIndex = 0;
        let iIndex = 0;
        let lastInd = 0;
        for (let context of SubmeshIterator(map.meshGroups)) {
            for (let shape of context.submesh.shapes) {
                for (let i = 0; i < shape.vtx.length; i++) {
                    vBuffer[vIndex++] = shape.vtx[i][0];
                    vBuffer[vIndex++] = shape.vtx[i][1];
                    vBuffer[vIndex++] = shape.vtx[i][2];
                    vBuffer[vIndex++] = shape.vn[i][0];
                    vBuffer[vIndex++] = shape.vn[i][1];
                    vBuffer[vIndex++] = shape.vn[i][2];
                    vBuffer[vIndex++] = shape.vcol[i][0];
                    vBuffer[vIndex++] = shape.vcol[i][1];
                    vBuffer[vIndex++] = shape.vcol[i][2];
                    vBuffer[vIndex++] = 1;
                    vBuffer[vIndex++] = shape.uv[i][0];
                    vBuffer[vIndex++] = shape.uv[i][1];
                }
                for (let i = 0; i < shape.ind.length; i++) {
                    iBuffer[iIndex++] = shape.ind[i] + lastInd;
                }
                lastInd += shape.vtx.length;
            }
        }
        this.vertexBuffer = makeStaticDataBuffer(device, GfxBufferUsage.VERTEX, vBuffer.buffer);
        this.indexBuffer = makeStaticDataBuffer(device, GfxBufferUsage.INDEX, iBuffer.buffer);

        const vertexAttributeDescriptors: GfxVertexAttributeDescriptor[] = [
            { location: SilentHill3Program.a_Position, bufferIndex: 0, format: GfxFormat.F32_RGB, bufferByteOffset: 0*0x04, },
            { location: SilentHill3Program.a_Normal, bufferIndex: 0, format: GfxFormat.F32_RGB, bufferByteOffset: 3*0x04, },
            { location: SilentHill3Program.a_Color, bufferIndex: 0, format: GfxFormat.F32_RGBA, bufferByteOffset: 6*0x04, },
            { location: SilentHill3Program.a_TexCoord, bufferIndex: 0, format: GfxFormat.F32_RG, bufferByteOffset: 10*0x04, },
        ];
        const vertexBufferDescriptors: GfxInputLayoutBufferDescriptor[] = [
            { byteStride: 12*0x04, frequency: GfxVertexBufferFrequency.PER_VERTEX, },
        ];
        this.inputLayout = device.createInputLayout({
            indexBufferFormat: GfxFormat.U32_R,
            vertexAttributeDescriptors,
            vertexBufferDescriptors
        });
        const buffers = [{ buffer: this.vertexBuffer, byteOffset: 0, byteStride: 12*4}];
        const indexBuffer = { buffer: this.indexBuffer, byteOffset: 0, byteStride: 0 };
        this.inputState = device.createInputState(this.inputLayout, buffers, indexBuffer);
    }

    public destroy(device: GfxDevice): void {
        device.destroySampler(this.sampler);
        device.destroyBuffer(this.indexBuffer);
        device.destroyBuffer(this.vertexBuffer);
        device.destroyInputLayout(this.inputLayout);
        device.destroyInputState(this.inputState);
    }
}

class SceneRenderer {
    private drawCallInstances: DrawCallInstance[] = [];
    private megaStateFlags: Partial<GfxMegaStateDescriptor>;

    constructor(sceneData: SceneData, drawCalls: DrawCall[], textureHolder: TextureHolder<any>) {
        this.megaStateFlags = {
            cullMode: GfxCullMode.BACK,
            depthWrite: true,
            depthCompare: reverseDepthForCompareMode(GfxCompareMode.LEQUAL),
        };

        for (let i = 0; i < drawCalls.length; i++) {
            this.drawCallInstances.push(new DrawCallInstance(sceneData, drawCalls[i], i, textureHolder));
        }
    }

    public prepareToRender(device: GfxDevice, renderInstManager: GfxRenderInstManager, viewerInput: Viewer.ViewerRenderInput) {
        const template = renderInstManager.pushTemplateRenderInst();
        template.setBindingLayouts(bindingLayouts);
        template.setMegaStateFlags(this.megaStateFlags);
        template.filterKey = RenderPass.MAIN;

        viewerInput.camera.setClipPlanes(/*n=*/20, /*f=*/500000);

        let offs = template.allocateUniformBuffer(SilentHill3Program.ub_SceneParams, 32);
        const sceneParamsMapped = template.mapUniformBufferF32(SilentHill3Program.ub_SceneParams);
        offs += fillMatrix4x4(sceneParamsMapped, offs, viewerInput.camera.projectionMatrix);
        offs += fillVec4(sceneParamsMapped, offs, viewerInput.camera.worldMatrix[12], viewerInput.camera.worldMatrix[13], viewerInput.camera.worldMatrix[14]);
        offs += fillVec4(sceneParamsMapped, offs, viewerInput.camera.viewMatrix[2], viewerInput.camera.viewMatrix[6], viewerInput.camera.viewMatrix[10]);
        sceneParamsMapped[offs] = viewerInput.time;
        
        for (const instance of this.drawCallInstances) {
            instance.prepareToRender(device, renderInstManager, viewerInput);
        }

        renderInstManager.popTemplateRenderInst();
    }

    public setVertexColorsEnabled(v: boolean): void {
        for (let i = 0; i < this.drawCallInstances.length; i++)
            this.drawCallInstances[i].setVertexColorsEnabled(v);
    }

    public setTexturesEnabled(v: boolean): void {
        for (let i = 0; i < this.drawCallInstances.length; i++)
            this.drawCallInstances[i].setTexturesEnabled(v);
    }
}

export class SilentHill3Renderer implements SceneGfx {
    private renderInstManager = new GfxRenderInstManager;
    private renderTarget = new BasicRenderTarget;
    private uniformBuffer: GfxRenderDynamicUniformBuffer;
    private sceneRenderers: SceneRenderer[] = [];
    private sceneData: SceneData;
    
    constructor(device: GfxDevice, map: SilentHill3Map, public textureHolder: TextureHolder<SilentHill3Texture>) {
        this.sceneData = new SceneData(device, map);
        this.uniformBuffer = new GfxRenderDynamicUniformBuffer(device);
        this.sceneRenderers.push(new SceneRenderer(this.sceneData, this.sceneData.drawCalls, this.textureHolder));
    }

    public createPanels(): UI.Panel[] {
        const renderHacksPanel = new UI.Panel();
        renderHacksPanel.customHeaderBackgroundColor = UI.COOL_BLUE_COLOR;
        renderHacksPanel.setTitle(UI.RENDER_HACKS_ICON, 'Render Hacks');
        const enableVertexColorsCheckbox = new UI.Checkbox('Enable Vertex Colors', true);
        enableVertexColorsCheckbox.onchanged = () => {
            this.sceneRenderers.forEach((sceneRenderer: SceneRenderer) => {
                sceneRenderer.setVertexColorsEnabled(enableVertexColorsCheckbox.checked);
            });
        };
        renderHacksPanel.contents.appendChild(enableVertexColorsCheckbox.elem);
        const enableTextures = new UI.Checkbox('Enable Textures', true);
        enableTextures.onchanged = () => {
            this.sceneRenderers.forEach((sceneRenderer: SceneRenderer) => {
                sceneRenderer.setTexturesEnabled(enableTextures.checked);
            });
        };
        renderHacksPanel.contents.appendChild(enableTextures.elem);
        return [renderHacksPanel];
    }

    public adjustCameraController(c: CameraController) {
        c.setSceneMoveSpeedMult(0.1);
    }

    protected prepareToRender(device: GfxDevice, hostAccessPass: GfxHostAccessPass, viewerInput: ViewerRenderInput) {
        const template = this.renderInstManager.pushTemplateRenderInst();
        template.setUniformBuffer(this.uniformBuffer);
        for (let i = 0; i < this.sceneRenderers.length; i++) {
            this.sceneRenderers[i].prepareToRender(device, this.renderInstManager, viewerInput);
        }
        this.renderInstManager.popTemplateRenderInst();
        this.uniformBuffer.prepareToRender(device, hostAccessPass);
    }

    public render(device: GfxDevice, viewerInput: ViewerRenderInput): GfxRenderPass {
        const hostAccessPass = device.createHostAccessPass();
        this.prepareToRender(device, hostAccessPass, viewerInput);
        device.submitPass(hostAccessPass);
        this.renderTarget.setParameters(device, viewerInput.backbufferWidth, viewerInput.backbufferHeight);

        // Create main render pass.
        const passRenderer = this.renderTarget.createRenderPass(device, viewerInput.viewport, transparentBlackFullClearRenderPassDescriptor);
        executeOnPass(this.renderInstManager, device, passRenderer, RenderPass.MAIN);
        this.renderInstManager.resetRenderInsts();
        return passRenderer;
    }

    public destroy(device: GfxDevice) {
        this.renderInstManager.destroy(device);
        this.uniformBuffer.destroy(device);
        this.renderTarget.destroy(device);
        this.textureHolder.destroy(device);
        this.sceneData.destroy(device);
    }
}

function textureToCanvas(texture: SilentHill3Texture): Viewer.Texture {
    const canvas = document.createElement("canvas");
    canvas.width = texture.width;
    canvas.height = texture.height;
    canvas.title = name;

    const ctx = canvas.getContext("2d")!;
    const imgData = ctx.createImageData(canvas.width, canvas.height);
    imgData.data.set(texture.pixels);
    ctx.putImageData(imgData, 0, 0);
    const surfaces = [canvas];

    const extraInfo = new Map<string, string>();
    extraInfo.set('Format', psmToString(texture.psm));

    return { name: texture.name, surfaces, extraInfo };
}

export class SilentHill3TextureHolder extends TextureHolder<SilentHill3Texture> {
    public addMapTextures(device: GfxDevice, map: SilentHill3Map) {
        this.addTextures(device, map.textures);
        this.addTextures(device, [map.dummyTexture]);
    }

    public loadTexture(device: GfxDevice, texture: SilentHill3Texture): LoadedTexture {
        const gfxTexture = device.createTexture(makeTextureDescriptor2D(GfxFormat.U8_RGBA_NORM, texture.width, texture.height, 1));
        device.setResourceName(gfxTexture, texture.name);
        const hostAccessPass = device.createHostAccessPass();
        hostAccessPass.uploadTextureData(gfxTexture, 0, [texture.pixels]);
        device.submitPass(hostAccessPass);

        const viewerTexture: Viewer.Texture = textureToCanvas(texture);
        return { gfxTexture, viewerTexture };
    }
}
