// Global type declarations for the DAN_G frontend

declare module 'react-plotly.js' {
  import { Component } from 'react';
  
  interface PlotParams {
    data: any[];
    layout?: any;
    config?: any;
    onInitialized?: (figure: any, graphDiv: HTMLElement) => void;
    onUpdate?: (figure: any, graphDiv: HTMLElement) => void;
    onPurge?: (figure: any, graphDiv: HTMLElement) => void;
    onError?: (err: any) => void;
    onBeforeHover?: (event: any) => void;
    onHover?: (event: any) => void;
    onUnhover?: (event: any) => void;
    onSelected?: (event: any) => void;
    onDeselect?: () => void;
    onRelayout?: (event: any) => void;
    onRedraw?: () => void;
    onAnimated?: () => void;
    onAfterExport?: () => void;
    onAfterPlot?: () => void;
    onAnimate?: () => void;
    onAnimatingFrame?: (event: any) => void;
    onAnimationInterrupted?: () => void;
    onAutoSize?: () => void;
    onBeforeExport?: () => void;
    onButtonClicked?: (event: any) => void;
    onClick?: (event: any) => void;
    onClickAnnotation?: (event: any) => void;
    onDblClick?: (event: any) => void;
    onFramework?: () => void;
    onLegendClick?: (event: any) => void;
    onLegendDoubleClick?: (event: any) => void;
    onRelayouting?: (event: any) => void;
    onRestyle?: (event: any) => void;
    onSelecting?: (event: any) => void;
    onSliderChange?: (event: any) => void;
    onSliderEnd?: (event: any) => void;
    onSliderStart?: (event: any) => void;
    onTransitioning?: () => void;
    onTransitionInterrupted?: () => void;
    onWebGlContextLost?: () => void;
    onWebGlContextRestored?: () => void;
    style?: React.CSSProperties;
    className?: string;
    divId?: string;
    debug?: boolean;
    useResizeHandler?: boolean;
    revision?: number;
  }

  class Plot extends Component<PlotParams> {}
  export default Plot;
}

declare module 'socket.io-client' {
  interface Socket {
    on(event: string, listener: (...args: any[]) => void): Socket;
    emit(event: string, ...args: any[]): Socket;
    connect(): Socket;
    disconnect(): Socket;
    close(): void;
    id: string;
    connected: boolean;
  }

  interface ManagerOptions {
    autoConnect?: boolean;
    reconnection?: boolean;
    reconnectionAttempts?: number;
    reconnectionDelay?: number;
    timeout?: number;
  }

  interface SocketOptions {
    auth?: any;
    query?: any;
  }

  function io(uri?: string, opts?: Partial<ManagerOptions & SocketOptions>): Socket;
  
  export default io;
  export { Socket, ManagerOptions, SocketOptions };
}

declare module 'react-hot-toast' {
  import { ReactNode } from 'react';
  
  interface ToastOptions {
    duration?: number;
    position?: 'top-left' | 'top-center' | 'top-right' | 'bottom-left' | 'bottom-center' | 'bottom-right';
    style?: React.CSSProperties;
    className?: string;
    icon?: ReactNode;
    iconTheme?: {
      primary: string;
      secondary: string;
    };
    ariaProps?: {
      role: string;
      'aria-live': 'polite' | 'assertive' | 'off';
    };
  }

  interface Toast {
    id: string;
    message: string;
    type: 'success' | 'error' | 'loading' | 'blank';
    duration?: number;
    createdAt: number;
    visible: boolean;
    height?: number;
    position: ToastOptions['position'];
  }

  interface ToasterProps {
    position?: ToastOptions['position'];
    toastOptions?: ToastOptions;
    reverseOrder?: boolean;
    gutter?: number;
    containerStyle?: React.CSSProperties;
    containerClassName?: string;
  }

  export const toast: {
    success: (message: string, options?: ToastOptions) => string;
    error: (message: string, options?: ToastOptions) => string;
    loading: (message: string, options?: ToastOptions) => string;
    custom: (jsx: ReactNode, options?: ToastOptions) => string;
    dismiss: (toastId?: string) => void;
    remove: (toastId?: string) => void;
    promise: <T>(
      promise: Promise<T>,
      msgs: {
        loading: string;
        success: string | ((data: T) => string);
        error: string | ((error: any) => string);
      },
      opts?: ToastOptions
    ) => Promise<T>;
  };

  export const Toaster: React.FC<ToasterProps>;
  export default toast;
}

declare module 'react-icons/fa' {
  import { IconType } from 'react-icons';
  
  export const FaUpload: IconType;
  export const FaChartLine: IconType;
  export const FaCogs: IconType;
  export const FaSyncAlt: IconType;
  export const FaExclamationTriangle: IconType;
  export const FaTachometerAlt: IconType;
  export const FaOilCan: IconType;
  export const FaSignOutAlt: IconType;
}

// Environment variables
declare namespace NodeJS {
  interface ProcessEnv {
    NEXT_PUBLIC_API_URL?: string;
    NEXTAUTH_URL?: string;
    NEXTAUTH_SECRET?: string;
  }
}
