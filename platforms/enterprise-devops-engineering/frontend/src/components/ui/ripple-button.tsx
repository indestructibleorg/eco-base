import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"

const rippleButtonVariants = cva(
  "relative inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 overflow-hidden",
  {
    variants: {
      variant: {
        default: "bg-[#F26207] text-white hover:bg-[#D95706] shadow-lg shadow-[#F26207]/25",
        destructive: "bg-red-500 text-white hover:bg-red-600",
        outline: "border border-[#2B3245] bg-transparent text-[#9AA0A6] hover:bg-[#2B3245] hover:text-white",
        secondary: "bg-[#1C2333] text-[#9AA0A6] hover:bg-[#2B3245] hover:text-white",
        ghost: "hover:bg-[#2B3245] hover:text-white text-[#9AA0A6]",
        link: "text-[#F26207] underline-offset-4 hover:underline",
        glow: "bg-[#F26207] text-white hover:bg-[#D95706] shadow-[0_0_20px_rgba(242,98,7,0.4)] hover:shadow-[0_0_30px_rgba(242,98,7,0.6)]",
      },
      size: {
        default: "h-10 px-4 py-2",
        sm: "h-8 rounded-md px-3 text-xs",
        lg: "h-12 rounded-md px-6 text-base",
        icon: "h-10 w-10",
        "icon-sm": "h-8 w-8",
        "icon-lg": "h-12 w-12",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

export interface RippleButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof rippleButtonVariants> {
  asChild?: boolean
  rippleColor?: string
  rippleDuration?: number
}

interface Ripple {
  id: number
  x: number
  y: number
  size: number
}

const RippleButton = React.forwardRef<HTMLButtonElement, RippleButtonProps>(
  ({ 
    className, 
    variant, 
    size, 
    asChild = false, 
    rippleColor = "rgba(255, 255, 255, 0.3)",
    rippleDuration = 600,
    onClick,
    children,
    ...props 
  }, ref) => {
    const [ripples, setRipples] = React.useState<Ripple[]>([])
    const buttonRef = React.useRef<HTMLButtonElement>(null)
    const rippleIdRef = React.useRef(0)

    const handleClick = (e: React.MouseEvent<HTMLButtonElement>) => {
      const button = buttonRef.current
      if (!button) return

      const rect = button.getBoundingClientRect()
      const x = e.clientX - rect.left
      const y = e.clientY - rect.top
      const size = Math.max(rect.width, rect.height) * 2

      const newRipple: Ripple = {
        id: rippleIdRef.current++,
        x,
        y,
        size,
      }

      setRipples((prev) => [...prev, newRipple])

      setTimeout(() => {
        setRipples((prev) => prev.filter((r) => r.id !== newRipple.id))
      }, rippleDuration)

      onClick?.(e)
    }

    const Comp = asChild ? Slot : "button"

    return (
      <Comp
        className={cn(rippleButtonVariants({ variant, size, className }))}
        ref={(node) => {
          // @ts-ignore
          buttonRef.current = node
          if (typeof ref === "function") {
            ref(node)
          } else if (ref) {
            ref.current = node
          }
        }}
        onClick={handleClick}
        {...props}
      >
        {children}
        {ripples.map((ripple) => (
          <span
            key={ripple.id}
            className="absolute rounded-full pointer-events-none animate-ripple"
            style={{
              left: ripple.x - ripple.size / 2,
              top: ripple.y - ripple.size / 2,
              width: ripple.size,
              height: ripple.size,
              backgroundColor: rippleColor,
              animationDuration: `${rippleDuration}ms`,
            }}
          />
        ))}
      </Comp>
    )
  }
)
RippleButton.displayName = "RippleButton"

export { RippleButton, rippleButtonVariants }
