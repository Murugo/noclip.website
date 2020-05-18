import * as PAC from './pac';
import * as UI from '../ui';
import * as Viewer from "../viewer";

// @ts-ignore
import program_glsl from './program.glsl';
import { BasicRenderTarget, transparentBlackFullClearRenderPassDescriptor } from '../gfx/helpers/RenderTargetHelpers';
import { GfxDevice, GfxHostAccessPass, GfxRenderPass, makeTextureDescriptor2D, GfxFormat, GfxMegaStateDescriptor, GfxCullMode, GfxBlendMode, GfxBlendFactor, GfxCompareMode, GfxBindingLayoutDescriptor, GfxBuffer, GfxInputLayout, GfxInputState, GfxSampler, GfxProgram, GfxWrapMode, GfxTexFilterMode, GfxMipFilterMode, GfxBufferUsage, GfxVertexAttributeDescriptor, GfxColorWriteMask, GfxInputLayoutBufferDescriptor, GfxVertexBufferFrequency } from '../gfx/platform/GfxPlatform';
import { GfxRenderInstManager, executeOnPass } from '../gfx/render/GfxRenderer';
import { GfxRenderDynamicUniformBuffer } from '../gfx/render/GfxRenderDynamicUniformBuffer';
import { TextureHolder, LoadedTexture, TextureMapping } from '../TextureHolder';
import { SceneGfx, ViewerRenderInput } from "../viewer";
import { psmToString } from '../Common/PS2/GS';
import { reverseDepthForCompareMode } from '../gfx/helpers/ReversedDepthHelpers';
import { fillMatrix4x4, fillMatrix4x3 } from '../gfx/helpers/UniformBufferHelpers';
import { DeviceProgram } from '../Program';
import { nArray,  } from '../util';
import { mat4, vec3 } from 'gl-matrix';
import { makeStaticDataBuffer } from '../gfx/helpers/BufferHelpers';

class HauntingGroundProgram extends DeviceProgram {
    public static a_Position = 0;
    public static a_Color = 1;
    public static a_TexCoord = 2;

    public static ub_SceneParams = 0;
    public static ub_DrawParams = 1;

    private static program = program_glsl;
    public both = HauntingGroundProgram.program;
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
    addBlendMode: boolean;
}

const worldScale = 25;
const modelMatrixScratch = mat4.fromScaling(mat4.create(), vec3.fromValues(worldScale, worldScale, worldScale));
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
        if (drawCall.addBlendMode) {
            this.megaStateFlags.attachmentsState = [
                {
                    colorWriteMask: GfxColorWriteMask.ALL,
                    rgbBlendState: {
                        blendMode: GfxBlendMode.ADD,
                        blendSrcFactor: GfxBlendFactor.SRC_ALPHA,
                        blendDstFactor: GfxBlendFactor.ONE,
                    },
                    alphaBlendState: {
                        blendMode: GfxBlendMode.ADD,
                        blendSrcFactor: GfxBlendFactor.ONE,
                        blendDstFactor: GfxBlendFactor.ZERO,
                    },
                }
            ];
            this.megaStateFlags.depthWrite = false;
        } else if (drawCall.translucent) {
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
            this.megaStateFlags.depthWrite = true;
            
            // Haunting Ground renders backfaces for all stage geometry, but this is
            // not practical for viewing stages. As a compromise, only turn off culling
            // for translucent draw calls.
            this.megaStateFlags.cullMode = GfxCullMode.NONE;
        }

        const textureMapping = this.textureMappings[0];
        const textureName = sceneData.textureNames[drawCall.textureIndex];
        if (drawCall.textureIndex >= 0 && textureHolder.hasTexture(textureName)) {
            textureHolder.fillTextureMapping(textureMapping, textureName);
        } else {
            this.textureMappings[0].gfxTexture = null;
        }
        textureMapping.gfxSampler = sceneData.sampler;
    }

    public prepareToRender(device: GfxDevice, renderInstManager: GfxRenderInstManager, viewerInput: Viewer.ViewerRenderInput) {
        const renderInst = renderInstManager.newRenderInst();
        renderInst.setInputLayoutAndState(this.sceneData.inputLayout, this.sceneData.inputState);
        renderInst.sortKey = this.drawCallIndex;
        if (this.gfxProgram === null) {
            this.gfxProgram = renderInstManager.gfxRenderCache.createProgram(device, this.program);
        }
        renderInst.setGfxProgram(this.gfxProgram);
        renderInst.setMegaStateFlags(this.megaStateFlags);
        renderInst.setSamplerBindingsFromTextureMappings(this.textureMappings);
        renderInst.drawIndexes(this.drawCall.indexCount, this.drawCall.firstIndex);

        let offs = renderInst.allocateUniformBuffer(HauntingGroundProgram.ub_DrawParams, 32);
        const mapped = renderInst.mapUniformBufferF32(HauntingGroundProgram.ub_DrawParams);
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
        const program = new HauntingGroundProgram();
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

interface RenderBatchKey {
    textureIndex: number;
    addBlendMode: boolean;
}

class SceneData {
    public vertexBuffer: GfxBuffer;
    public indexBuffer: GfxBuffer;
    public inputLayout: GfxInputLayout;
    public inputState: GfxInputState;

    public drawCalls: DrawCall[] = [];
    public sampler: GfxSampler;

    public textureNames: string[] = [];

    private vertices = 0;
    private indices = 0;

    constructor(device: GfxDevice, stage: PAC.HauntingGroundStage) {
        for (let i = 0; i < stage.textures.length; i++) {
            this.textureNames[i] = stage.textures[i].name;
        }
        this.sampler = this.createSampler(device);

        const meshesInDrawOrder: PAC.MeshGroup[] = [];
        this.createBatchedDrawCalls(stage, meshesInDrawOrder);
        this.buildBuffersAndInputState(device, meshesInDrawOrder);
    }

    private createSampler(device: GfxDevice) {
        return device.createSampler({
            wrapS: GfxWrapMode.REPEAT,
            wrapT: GfxWrapMode.REPEAT,
            minFilter: GfxTexFilterMode.BILINEAR,
            magFilter: GfxTexFilterMode.BILINEAR,
            mipFilter: GfxMipFilterMode.NO_MIP,
            minLOD: 0, maxLOD: 0,
        });
    }

    private createBatchedDrawCalls(stage: PAC.HauntingGroundStage, meshesInDrawOrder: PAC.MeshGroup[]) {
        const opaqueMeshMap: Map<string, PAC.MeshGroup[]> = new Map;
        const translucentMeshes: Array<[string, PAC.MeshGroup[]]> = [];

        let lastIndex = -1;
        for (const meshGroup of stage.meshGroups) {
            if (meshGroup.textureIndex < 0) {
                // Mesh used for lights. Typically does not have UVs.
                // TODO: Add support to render these correctly.
                continue;
            }
            const batchKey: RenderBatchKey = { textureIndex: meshGroup.textureIndex, addBlendMode: meshGroup.addBlendMode };
            const batchKeyStr = JSON.stringify(batchKey);
            if (!meshGroup.translucent) {
                if (!opaqueMeshMap.has(batchKeyStr)) {
                    opaqueMeshMap.set(batchKeyStr, []);
                }
                opaqueMeshMap.get(batchKeyStr)!.push(meshGroup);
            } else if (lastIndex >= 0 && translucentMeshes[lastIndex][0] === batchKeyStr) {
                translucentMeshes[lastIndex][1].push(meshGroup);
            } else {
                translucentMeshes.push([batchKeyStr, [meshGroup]])
                lastIndex++;
            }
        }

        opaqueMeshMap.forEach((value: PAC.MeshGroup[], key: string) => {
            const batchKey: RenderBatchKey = JSON.parse(key);
            const firstIndex = this.indices;
            for (const meshGroup of value) {
                meshesInDrawOrder.push(meshGroup);
                this.vertices += meshGroup.vtx.length;
                this.indices += meshGroup.ind.length;
            }
            const indexCount = this.indices - firstIndex;
            this.drawCalls.push({ textureIndex: batchKey.textureIndex, firstIndex, indexCount, translucent: false, addBlendMode: batchKey.addBlendMode });
        });
        translucentMeshes.forEach((value: [string, PAC.MeshGroup[]]) => {
            const batchKey: RenderBatchKey = JSON.parse(value[0]);
            const firstIndex = this.indices;
            for (const meshGroup of value[1]) {
                meshesInDrawOrder.push(meshGroup);
                this.vertices += meshGroup.vtx.length;
                this.indices += meshGroup.ind.length;
            }
            const indexCount = this.indices - firstIndex;
            this.drawCalls.push({ textureIndex: batchKey.textureIndex, firstIndex, indexCount, translucent: true, addBlendMode: batchKey.addBlendMode });
        });
    }

    private buildBuffersAndInputState(device: GfxDevice, meshesInDrawOrder: PAC.MeshGroup[]) {
        const vBuffer = new Float32Array(this.vertices * 9);
        const iBuffer = new Uint32Array(this.indices);
        let vIndex = 0;
        let iIndex = 0;
        let lastInd = 0;
        for (const meshGroup of meshesInDrawOrder) {
            for (let i = 0; i < meshGroup.vtx.length; i++) {
                vBuffer[vIndex++] = meshGroup.vtx[i][0];
                vBuffer[vIndex++] = meshGroup.vtx[i][1];
                vBuffer[vIndex++] = meshGroup.vtx[i][2];
                vBuffer[vIndex++] = meshGroup.vcol[i][0];
                vBuffer[vIndex++] = meshGroup.vcol[i][1];
                vBuffer[vIndex++] = meshGroup.vcol[i][2];
                vBuffer[vIndex++] = meshGroup.vcol[i][3];
                vBuffer[vIndex++] = meshGroup.uv[i][0];
                vBuffer[vIndex++] = meshGroup.uv[i][1];
            }
            for (let i = 0; i < meshGroup.ind.length; i++) {
                iBuffer[iIndex++] = meshGroup.ind[i] + lastInd;
            }
            lastInd += meshGroup.vtx.length;
        }

        this.vertexBuffer = makeStaticDataBuffer(device, GfxBufferUsage.VERTEX, vBuffer.buffer);
        this.indexBuffer = makeStaticDataBuffer(device, GfxBufferUsage.INDEX, iBuffer.buffer);

        const vertexAttributeDescriptors: GfxVertexAttributeDescriptor[] = [
            { location: HauntingGroundProgram.a_Position, bufferIndex: 0, format: GfxFormat.F32_RGB, bufferByteOffset: 0*0x04, },
            { location: HauntingGroundProgram.a_Color, bufferIndex: 0, format: GfxFormat.F32_RGBA, bufferByteOffset: 3*0x04, },
            { location: HauntingGroundProgram.a_TexCoord, bufferIndex: 0, format: GfxFormat.F32_RG, bufferByteOffset: 7*0x04, },
        ];
        const vertexBufferDescriptors: GfxInputLayoutBufferDescriptor[] = [
            { byteStride: 9*0x04, frequency: GfxVertexBufferFrequency.PER_VERTEX, },
        ];
        this.inputLayout = device.createInputLayout({
            indexBufferFormat: GfxFormat.U32_R,
            vertexAttributeDescriptors,
            vertexBufferDescriptors
        });
        const buffers = [{ buffer: this.vertexBuffer, byteOffset: 0, byteStride: 9*4}];
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

        let offs = template.allocateUniformBuffer(HauntingGroundProgram.ub_SceneParams, 20);
        const sceneParamsMapped = template.mapUniformBufferF32(HauntingGroundProgram.ub_SceneParams);
        offs += fillMatrix4x4(sceneParamsMapped, offs, viewerInput.camera.projectionMatrix);
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

export class HauntingGroundRenderer implements SceneGfx {
    private renderInstManager = new GfxRenderInstManager;
    private renderTarget = new BasicRenderTarget;
    private uniformBuffer: GfxRenderDynamicUniformBuffer;
    private sceneRenderers: SceneRenderer[] = [];
    private sceneData: SceneData;

    constructor(device: GfxDevice, stage: PAC.HauntingGroundStage, public textureHolder: TextureHolder<any>) {
        this.sceneData = new SceneData(device, stage);
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

function textureToCanvas(texture: PAC.Texture): Viewer.Texture {
    const canvas = document.createElement("canvas");
    const width = texture.width;
    const height = texture.height;
    const name = texture.name;
    canvas.width = width;
    canvas.height = height;
    canvas.title = name;

    const ctx = canvas.getContext("2d")!;
    const imgData = ctx.createImageData(canvas.width, canvas.height);
    imgData.data.set(texture.pixels);
    ctx.putImageData(imgData, 0, 0);
    const surfaces = [canvas];

    const extraInfo = new Map<string, string>();
    extraInfo.set('Format', psmToString(texture.psm));

    return { name: name, surfaces, extraInfo };
}

export class HauntingGroundTextureHolder extends TextureHolder<PAC.Texture> {
    public addStageTextures(device: GfxDevice, stage: PAC.HauntingGroundStage) {
        this.addTextures(device, stage.textures);
    }

    public loadTexture(device: GfxDevice, texture: PAC.Texture): LoadedTexture {
        const gfxTexture = device.createTexture(makeTextureDescriptor2D(GfxFormat.U8_RGBA_NORM, texture.width, texture.height, 1));
        device.setResourceName(gfxTexture, texture.name);
        const hostAccessPass = device.createHostAccessPass();
        hostAccessPass.uploadTextureData(gfxTexture, 0, [texture.pixels]);
        device.submitPass(hostAccessPass);

        const viewerTexture: Viewer.Texture = textureToCanvas(texture);
        return { gfxTexture, viewerTexture };
    }
}
