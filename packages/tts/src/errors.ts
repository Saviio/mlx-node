/** The reference input cannot be processed by this checkpoint. */
export class InvalidReferenceAudioError extends Error {
  override readonly name = 'InvalidReferenceAudioError';
}
